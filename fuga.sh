for tester_name in "pixartsigma"; do
    echo "Running ${tester_name}"
    poetry run python -m opencole.inference_launcher.gpt4_augmented_baselines --tester ${tester_name} --output_dir tmp/results/designerintentionv2/${tester_name} --split_name designerintention_v2
done
