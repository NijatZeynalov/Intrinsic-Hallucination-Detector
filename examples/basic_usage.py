import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from src.detection.dynamic_detector import DynamicHallucationDetector
from src.detection.error_classifier import ErrorClassifier
from src.utils.logger import get_logger
from src.config.config import DetectorConfig

logger = get_logger(__name__)


def main():
    """Basic example of using the Hallucination Detector."""
    try:
        # Initialize model and tokenizer
        model_name = "meta-llama/Llama-2-7b"
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Initialize detector
        config = DetectorConfig()
        detector = DynamicHallucationDetector(config)

        # Example prompts
        prompts = [
            "Explain quantum computing in simple terms.",
            "What is the history of the moon landing?",
            "Describe the process of photosynthesis."
        ]

        # Process prompts
        for prompt in prompts:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate
            outputs = model.generate(
                inputs.input_ids,
                max_length=200,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True
            )

            # Analyze for hallucinations
            results = detector.detect(
                outputs.hidden_states[-1],
                outputs.attentions[-1],
                outputs.sequences
            )

            # Print results
            print(f"\nPrompt: {prompt}")
            print(f"Hallucination probability: {results['hallucination_probs'].mean():.2f}")
            print(f"Confidence score: {results['confidence_scores'].mean():.2f}")
            print("-" * 50)

    except Exception as e:
        logger.error(f"Error in basic usage: {str(e)}")


if __name__ == "__main__":
    main()