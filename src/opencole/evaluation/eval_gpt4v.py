import argparse
import base64
import logging
import os
import time
from mimetypes import guess_type
from pathlib import Path

import pandas as pd
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel, validator
from openai import AzureOpenAI
from PIL import Image, UnidentifiedImageError

MAX_N_RETRY = 10
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

QUALITY_ASSURANCE_PROMPT = (
    "You are an autonomous AI Assistant who aids designers by providing insightful, objective, and constructive critiques of graphic design projects. "
    + "Your goals are: Deliver comprehensive and unbiased evaluations of graphic designs based on established design principles and industry standards. "
    + "You are asked to choose the design that best matches a given intention. "
    + "Identify potential areas for improvement and suggest actionable feedback to enhance the overall aesthetic and effectiveness of the designs. "
    + "Maintain a consistent and high standard of critique. "
    + "Utilize coordinate information for data description relative to the upper left corner of the image, with the upper left corner serving as the origin, the right as the positive direction, and the downward as the positive direction. "
    + "Please abide by the following rules: Strive to answer as objectively as possible. "
    + "Grade seriously."
    + "If the output is too long, it will be truncated. "
    + "Only respond in JSON format, no other information. "
    + 'Example of output for designs that most match a given intention: {"best_design": "b", explanation: "(Please explain the reason of choice.)"}\n'
)


def main() -> None:
    """
    Evaluate the final design using Azure OpenAI GPT-4 V.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Flat directory containing images."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="csv containing id and scores are generated.",
    )
    parser.add_argument("--sleep_sec", type=float, default=5.0)
    args = parser.parse_args()
    logger.info(f"{args=}")

    output_dir = Path(args.output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    parser = PydanticOutputParser(pydantic_object=Result)

    results: list[dict] = []

    for i, image_path in enumerate(sorted(Path(args.input_dir).glob("*.*"))):
        try:
            Image.open(image_path)
        except UnidentifiedImageError:
            # `image` is not image-like data
            continue

        logger.info(f"Querying {image_path=}")
        n_retry = 0
        while True:
            if n_retry >= MAX_N_RETRY:
                # get out of the loop by marking as failed.
                output = Result(**{k: -1 for k in Result.__fields__})
                break
            try:
                response = client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_GPT4V_DEPLOYMENT_NAME"),
                    messages=[
                        {"role": "system", "content": QUALITY_ASSURANCE_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Which of the following images better matches the intention?",
                                },
                                {"type": "text", "text": "(a)"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": local_image_to_data_url(str(image_path)),
                                        "detail": "low",
                                    },
                                },
                                {"type": "text", "text": "(b)"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": local_image_to_data_url(
                                            str(image_path)
                                        ),
                                        "detail": "low",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=2000,
                )

                logger.info(response)

                finish_reason = response.choices[0].finish_reason
                if finish_reason not in ["stop", None]:
                    logger.error(f"Unexpected {finish_reason=}")
                    n_retry += 1
                    time.sleep(args.sleep_sec)
                    continue

                content = response.choices[0].message.content
                output = parser.parse(content)
            except OutputParserException:
                logger.error(f"Failed to parse {output=}")
                n_retry += 1
                time.sleep(args.sleep_sec)
                continue
            except Exception as e:
                logger.error(f"Unexpected exception {e=}")
                n_retry += 1
                time.sleep(args.sleep_sec)
                continue

            # if the script reaches here, the parsing was successful
            break

        results.append({"id": image_path.stem, **output.dict()})

        if os.environ.get("LOGLEVEL", "INFO") == "DEBUG" and i >= 4:
            # -1 is for 0-based index
            break

        time.sleep(args.sleep_sec)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(args.output_path, index=False)


class Result(BaseModel):  # type: ignore
    """
    Result of the evaluation
    """

    design_and_layout: int
    content_relevance_and_effectiveness: int
    typography_and_color_scheme: int
    graphics_and_images: int
    innovation_and_originality: int

    @validator(
        "design_and_layout",
        "content_relevance_and_effectiveness",
        "typography_and_color_scheme",
        "graphics_and_images",
        "innovation_and_originality",
    )
    def validate_score(cls, v: int) -> int:
        if not ((1 <= v <= 10) or v == -1):
            raise ValueError(
                "Score must be between 1 and 10, or -1 (failed to evaluate)"
            )
        return v


def local_image_to_data_url(image_path: str) -> str:
    """
    Function to encode a local image into data URL
    """
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


if __name__ == "__main__":
    main()
