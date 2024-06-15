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
from openai import AzureOpenAI, OpenAI
from PIL import Image, UnidentifiedImageError

MAX_N_RETRY = 10
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


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

    # client = AzureOpenAI(
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    # )
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
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
                                {"type": "text", "text": "Grade this picture."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": local_image_to_data_url(str(image_path)),
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


QUALITY_ASSURANCE_PROMPT = (
    "You are an autonomous AI Assistant who aids designers by providing insightful, objective, and constructive critiques of graphic design projects. "
    "Your goals are: Deliver comprehensive and unbiased evaluations of graphic designs based on established design principles and industry standards. "
    "Identify potential areas for improvement and suggest actionable feedback to enhance the overall aesthetic and effectiveness of the designs. "
    "Maintain a consistent and high standard of critique. "
    "Utilize coordinate information for data description relative to the upper left corner of the image, with the upper left corner serving as the origin, the right as the positive direction, and the downward as the positive direction. "
    "Please abide by the following rules: Strive to score as objectively as possible. "
    "Grade seriously. A flawless design can earn 10 points, a mediocre design can only earn 7 points, a design with obvious shortcomings can only earn 4 points, and a very poor design can only earn 1-2 points. "
    "Keep your reasoning concise when rating, and describe it as briefly as possible."
    "If the output is too long, it will be truncated. "
    "Only respond in JSON format, no other information. "
    'Example of output for a perfect design: {"design_and_layout": 10, "content_relevance_and_effectiveness": 10, "typography_and_color_scheme": 10, "graphics_and_images": 10, "innovation_and_originality": 10} '
    "Grading criteria\n"
    "Design and Layout (name: design_and_layout, range: 1-10): The graphic design should present a clean, balanced, and consistent layout. "
    "The organization of elements should enhance the message, with clear paths for the eye to follow. "
    "A score of 10 signifies a layout that maximizes readability and visual appeal, while a 1 indicates a cluttered, confusing layout with no clear hierarchy or flow. "
    "Content Relevance and Effectiveness (name: content_relevance_and_effectiveness, range: 1-10): The content should be not only relevant to its purpose but also engaging for the intended audience, effectively communicating the intended message."
    "A score of 10 means the content resonates with the target audience, aligns with the design’s purpose, and enhances the overall message. "
    "A score of 1 indicates the content is irrelevant or does not connect with the audience. "
    "Typography and Color Scheme (name: typography_and_color_scheme, range: 1-10): Typography and color should work together to enhance readability and harmonize with other design elements."
    "This includes font selection, size, line spacing, color, and placement, as well as the overall color scheme of the design. "
    "A score of 10 represents excellent use of typography and color that aligns with the design’s purpose and aesthetic, while a score of 1 indicates poor use of these elements that hinders readability or clashes with the design."
    "Graphics and Images (name: graphics_and_images, range: 1-10): Any graphics or images used should enhance the design rather than distract from it. They should be high quality, relevant, and harmonious with other elements. "
    "A score of 10 indicates graphics or images that enhance the overall design and message, while a 1 indicates low-quality, irrelevant, or distracting visuals."
    "Innovation and Originality (name: innovation_and_originality, range: 1-10): The design should display an original, creative approach. It should not just follow trends but also show a unique interpretation of the brief. "
    "A score of 10 indicates a highly creative and innovative design that stands out in its originality, while a score of 1 indicates a lack of creativity or a generic approach."
)


if __name__ == "__main__":
    main()
