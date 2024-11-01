import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from src.visualization.state_visualizer import StateVisualizer
from src.visualization.activation_plotter import ActivationPlotter
from src.config.config import VisualizationConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Demo of visualization capabilities."""
    try:
        # Initialize
        model_name = "meta-llama/Llama-2-7b"
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Initialize visualizers
        config = VisualizationConfig()
        state_viz = StateVisualizer(config)
        act_plotter = ActivationPlotter(config)

        # Example text
        text = "The speed of light in vacuum is exactly 299,792 kilometers per second."
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate with attention and states
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True
        )

        # Create visualizations
        state_viz.visualize_states(
            outputs.hidden_states[-1],
            tokenizer.convert_ids_to_tokens(outputs.sequences[0]),
            save_path="states.html"
        )

        act_plotter.plot_layer_activations(
            outputs.hidden_states[-1],
            outputs.attentions[-1],
            save_path="activations.html"
        )

        logger.info("Visualizations saved successfully")

    except Exception as e:
        logger.error(f"Error in visualization demo: {str(e)}")

if __name__ == "__main__":
    main()