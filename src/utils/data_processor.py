import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizer
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Handles data processing for hallucination detection, including
    data preparation, batching, and preprocessing.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_length: int = 512,
            batch_size: int = 32,
            cache_dir: Optional[str] = "cache"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track processed data statistics
        self.stats = defaultdict(int)
        self.cached_data = {}

    def prepare_input_sequence(
            self,
            text: Union[str, List[str]],
            labels: Optional[Union[str, List[str]]] = None,
            return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input sequences for the model.

        Args:
            text: Input text or list of texts
            labels: Optional ground truth text(s)
            return_tensors: Whether to return PyTorch tensors

        Returns:
            Dict containing processed inputs
        """
        try:
            # Handle single text input
            if isinstance(text, str):
                text = [text]
            if isinstance(labels, str):
                labels = [labels]

            # Tokenize inputs
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt" if return_tensors else None
            )

            # Process labels if provided
            if labels:
                label_tokens = self.tokenizer(
                    labels,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt" if return_tensors else None
                )
                inputs["labels"] = label_tokens["input_ids"]

            # Track statistics
            self.stats["processed_sequences"] += len(text)
            self.stats["total_tokens"] += inputs["input_ids"].size(1)

            return inputs

        except Exception as e:
            logger.error(f"Error preparing input sequence: {str(e)}")
            raise

    def create_attention_masks(
            self,
            input_ids: torch.Tensor,
            sliding_window: int = 128,
            overlap: int = 64
    ) -> torch.Tensor:
        """
        Create attention masks for long sequences using sliding windows.

        Args:
            input_ids: Input token IDs
            sliding_window: Size of attention window
            overlap: Overlap between windows

        Returns:
            torch.Tensor: Attention masks
        """
        batch_size, seq_length = input_ids.shape

        # Create base attention mask
        attention_mask = torch.zeros(
            (batch_size, seq_length, seq_length),
            device=input_ids.device
        )

        # Apply sliding windows
        for start in range(0, seq_length, sliding_window - overlap):
            end = min(start + sliding_window, seq_length)

            # Allow attention within window
            attention_mask[:, start:end, start:end] = 1

            # Add overlap attention
            if start > 0:
                overlap_start = max(0, start - overlap)
                attention_mask[:, start:end, overlap_start:start] = 1
                attention_mask[:, overlap_start:start, start:end] = 1

        return attention_mask

    def prepare_evaluation_batch(
            self,
            texts: List[str],
            references: Optional[List[str]] = None,
            include_token_info: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for hallucination evaluation.

        Args:
            texts: List of texts to evaluate
            references: Optional reference texts
            include_token_info: Whether to include token information

        Returns:
            Dict containing processed batch
        """
        batch = {}

        # Process inputs
        inputs = self.prepare_input_sequence(texts)
        batch.update(inputs)

        # Add token information if requested
        if include_token_info:
            token_info = self._get_token_information(
                inputs["input_ids"],
                texts
            )
            batch.update(token_info)

        # Process references if provided
        if references:
            ref_inputs = self.prepare_input_sequence(references)
            batch["reference_ids"] = ref_inputs["input_ids"]

            # Add reference token information
            if include_token_info:
                ref_token_info = self._get_token_information(
                    ref_inputs["input_ids"],
                    references
                )
                batch["reference_token_info"] = ref_token_info

        return batch

    def _get_token_information(
            self,
            input_ids: torch.Tensor,
            texts: List[str]
    ) -> Dict[str, List]:
        """Get detailed token information."""
        token_info = {
            "token_texts": [],
            "token_positions": [],
            "token_types": []
        }

        for i, text in enumerate(texts):
            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids[i].tolist()
            )

            # Get token texts
            token_info["token_texts"].append(tokens)

            # Get token positions
            positions = list(range(len(tokens)))
            token_info["token_positions"].append(positions)

            # Get token types (word start, continuation, special)
            types = []
            for token in tokens:
                if token in self.tokenizer.all_special_tokens:
                    types.append("special")
                elif token.startswith("▁"):  # For SentencePiece tokenizers
                    types.append("word_start")
                else:
                    types.append("continuation")
            token_info["token_types"].append(types)

        return token_info

    def create_batches(
            self,
            data: List[Dict],
            shuffle: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create batches from data.

        Args:
            data: List of data items
            shuffle: Whether to shuffle the data

        Returns:
            List of batches
        """
        if shuffle:
            np.random.shuffle(data)

        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_data = data[i:i + self.batch_size]
            batch = self._collate_batch(batch_data)
            batches.append(batch)

        return batches

    def _collate_batch(
            self,
            batch_data: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Collate batch items."""
        batch = defaultdict(list)

        for item in batch_data:
            for key, value in item.items():
                batch[key].append(value)

        # Convert to tensors
        for key in batch:
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.stack(batch[key])
            elif isinstance(batch[key][0], (int, float)):
                batch[key] = torch.tensor(batch[key])

        return dict(batch)

    def cache_data(
            self,
            data: Dict[str, torch.Tensor],
            cache_key: str
    ) -> None:
        """
        Cache processed data.

        Args:
            data: Data to cache
            cache_key: Cache identifier
        """
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pt"
            torch.save(data, cache_file)
            self.cached_data[cache_key] = True

    def load_cached_data(
            self,
            cache_key: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load cached data.

        Args:
            cache_key: Cache identifier

        Returns:
            Optional[Dict]: Cached data if available
        """
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.pt"
            if cache_file.exists():
                return torch.load(cache_file)
        return None

    def get_statistics(self) -> Dict:
        """Get data processing statistics."""
        stats = {
            "processed_sequences": self.stats["processed_sequences"],
            "total_tokens": self.stats["total_tokens"],
            "average_sequence_length": (
                    self.stats["total_tokens"] / max(1, self.stats["processed_sequences"])
            ),
            "cached_items": len(self.cached_data)
        }
        return stats

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats.clear()


class HallucationDataProcessor(DataProcessor):
    """
    Specialized data processor for hallucination detection tasks.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_length: int = 512,
            batch_size: int = 32,
            cache_dir: Optional[str] = "cache"
    ):
        super().__init__(tokenizer, max_length, batch_size, cache_dir)
        self.special_tokens = self._get_special_tokens()

    def _get_special_tokens(self) -> List[str]:
        """Get relevant special tokens for hallucination detection."""
        return [
            self.tokenizer.pad_token,
            self.tokenizer.eos_token,
            self.tokenizer.bos_token,
            self.tokenizer.unk_token
        ]

    def prepare_analysis_sequence(
            self,
            source_text: str,
            generated_text: str,
            reference_text: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare sequences for hallucination analysis.

        Args:
            source_text: Input prompt text
            generated_text: Generated model output
            reference_text: Optional reference text

        Returns:
            Dict containing processed sequences
        """
        # Process source and generated text
        inputs = self.prepare_input_sequence(
            [source_text, generated_text],
            return_tensors=True
        )

        # Add sequence markers
        batch = {
            "source_ids": inputs["input_ids"][0],
            "generated_ids": inputs["input_ids"][1],
            "source_mask": inputs["attention_mask"][0],
            "generated_mask": inputs["attention_mask"][1]
        }

        # Add reference if provided
        if reference_text:
            ref_inputs = self.prepare_input_sequence(
                [reference_text],
                return_tensors=True
            )
            batch["reference_ids"] = ref_inputs["input_ids"][0]
            batch["reference_mask"] = ref_inputs["attention_mask"][0]

        # Add token analysis
        batch.update(self._analyze_tokens(batch))

        return batch

    def _analyze_tokens(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> Dict[str, List]:
        """Analyze tokens for potential hallucination indicators."""
        analysis = {}

        # Get token strings
        source_tokens = self.tokenizer.convert_ids_to_tokens(
            batch["source_ids"]
        )
        generated_tokens = self.tokenizer.convert_ids_to_tokens(
            batch["generated_ids"]
        )

        # Analyze token overlap
        overlap_indices = self._find_token_overlap(
            batch["source_ids"],
            batch["generated_ids"]
        )

        analysis.update({
            "source_tokens": source_tokens,
            "generated_tokens": generated_tokens,
            "token_overlap": overlap_indices,
            "special_token_positions": self._find_special_tokens(
                batch["generated_ids"]
            )
        })

        return analysis

    def _find_token_overlap(
            self,
            source_ids: torch.Tensor,
            generated_ids: torch.Tensor
    ) -> List[int]:
        """Find overlapping tokens between source and generated text."""
        overlap_indices = []
        source_set = set(source_ids.tolist())

        for idx, token_id in enumerate(generated_ids):
            if token_id.item() in source_set:
                overlap_indices.append(idx)

        return overlap_indices

    def _find_special_tokens(
            self,
            input_ids: torch.Tensor
    ) -> List[int]:
        """Find positions of special tokens."""
        special_positions = []
        special_ids = set(
            self.tokenizer.convert_tokens_to_ids(self.special_tokens)
        )

        for idx, token_id in enumerate(input_ids):
            if token_id.item() in special_ids:
                special_positions.append(idx)

        return special_positions

    def get_token_importances(
            self,
            attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate token importance scores based on attention.

        Args:
            attention_weights: Attention matrix

        Returns:
            torch.Tensor: Token importance scores
        """
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=1)

        # Calculate token importance as attention sum
        importance_scores = avg_attention.sum(dim=-1)

        return importance_scores