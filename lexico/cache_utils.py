from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformers.cache_utils import DynamicCache, CacheConfig

from lexico.omp import omp

@dataclass
class LexicoCacheConfig(CacheConfig):
    def __init__(
        self,
        key_max_sparsity: Optional[int] = 14,
        value_max_sparsity: Optional[int] = 14,
        key_error_threshold: Optional = None,
        value_error_threshold: Optional = None,
        buffer_length: Optional[int] = 128,
        approximation_length: Optional[int] = 32,
    ):
        self.key_max_sparsity = key_max_sparsity
        self.value_max_sparsity = value_max_sparsity
        self.key_error_threshold = key_error_threshold
        self.value_error_threshold = value_error_threshold
        self.buffer_length = buffer_length
        self.approximation_length = approximation_length
    
    def validate(self):
        """Validates if the arguments passed are correct"""

        incorrect_arg_msg = (
            "Some of the keys in `cache_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        # Check that the values are reasonable in general
        if self.buffer_length <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="residual_length",
                    correct_value="a positive integer",
                    found_value=self.residual_length,
                ),
            )
        if self.approximation_length <= 0:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="approximation_length",
                    correct_value="a positive integer",
                    found_value=self.approximation_length,
                ),
            )
        if self.approximation_length > self.buffer_length:
            raise ValueError(
                incorrect_arg_msg.format(
                    key="approximation_length",
                    correct_value="larger than the buffer length",
                    found_value=self.approximation_length,
                ),
            )


class LexicoCache(DynamicCache):
    def __init__(self, cache_config: LexicoCacheConfig) -> None:
        super().__init__()
        self._key_cache_crow_indices: List[torch.Tensor] = []
        self._key_cache_col_indices: List[torch.Tensor] = []
        self._key_cache_values: List[torch.Tensor] = []
        self._value_cache_crow_indices: List[torch.Tensor] = []
        self._value_cache_col_indices: List[torch.Tensor] = []
        self._value_cache_values: List[torch.Tensor] = []

        self.key_max_sparsity = cache_config.key_max_sparsity
        self.value_max_sparsity = cache_config.value_max_sparsity
        self.key_error_threshold = cache_config.key_error_threshold
        self.value_error_threshold = cache_config.value_error_threshold
        self.buffer_length = cache_config.buffer_length
        self.approximation_length = cache_config.approximation_length
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        key_dictionary: torch.Tensor,
        value_dictionary: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        if len(self.key_cache) < layer_idx:
            raise ValueError
        elif len(self.key_cache) == layer_idx:
            if key_states.shape[-2] > self.buffer_length:
                keys_crow_indices, keys_col_indices, keys_values = self._compress(key_states[:, :, :-self.buffer_length, :], key_dictionary, self.key_max_sparsity, self.key_error_threshold)
                values_crow_indices, values_col_indices, values_values = self._compress(value_states[:, :, :-self.buffer_length, :], value_dictionary, self.value_max_sparsity, self.value_error_threshold)
                
                self._key_cache_crow_indices.append(keys_crow_indices)
                self._key_cache_col_indices.append(keys_col_indices)
                self._key_cache_values.append(keys_values)
                self._value_cache_crow_indices.append(values_crow_indices)
                self._value_cache_col_indices.append(values_col_indices)
                self._value_cache_values.append(values_values)
            else:
                self._key_cache_crow_indices.append(torch.tensor([0], dtype=torch.int32, device=key_states.device))
                self._key_cache_col_indices.append(torch.empty(0, device=key_states.device))
                self._key_cache_values.append(torch.empty(0, device=key_states.device))
                self._value_cache_crow_indices.append(torch.tensor([0], dtype=torch.int32, device=value_states.device))
                self._value_cache_col_indices.append(torch.empty(0, device=value_states.device))
                self._value_cache_values.append(torch.empty(0, device=value_states.device))

            self.key_cache.append(key_states[:, :, -self.buffer_length:, :])
            self.value_cache.append(value_states[:, :, -self.buffer_length:, :])
            return None, key_states, None, value_states
        else:
            if len(self._key_cache_crow_indices[layer_idx]) > 1:
                compressed_key_states_to_return = torch.sparse_csr_tensor(
                    self._key_cache_crow_indices[layer_idx].to(torch.int32),
                    self._key_cache_col_indices[layer_idx].to(torch.int32),
                    self._key_cache_values[layer_idx].to(torch.float16),
                    size=(self._key_cache_crow_indices[layer_idx].shape[0]-1, key_dictionary.shape[1])
                )
            else:
                compressed_key_states_to_return = None
            if len(self._value_cache_crow_indices[layer_idx]) > 1:
                compressed_value_states_to_return = torch.sparse_csr_tensor(
                    self._value_cache_crow_indices[layer_idx].to(torch.int32),
                    self._value_cache_col_indices[layer_idx].to(torch.int32),
                    self._value_cache_values[layer_idx].to(torch.float16),
                    size=(self._value_cache_crow_indices[layer_idx].shape[0]-1, value_dictionary.shape[1])
                )
            else:
                compressed_value_states_to_return = None
            
            keys_to_return = [self.key_cache[layer_idx], key_states]
            values_to_return = [self.value_cache[layer_idx], value_states]

            keys_to_return = torch.cat(keys_to_return, dim=-2)
            values_to_return = torch.cat(values_to_return, dim=-2)

            if self.key_cache[layer_idx].shape[-2] >= self.buffer_length:
                keys_crow_indices, keys_col_indices, keys_values = self._compress(self.key_cache[layer_idx][:, :, :self.approximation_length], key_dictionary, self.key_max_sparsity, self.key_error_threshold)
                values_crow_indices, values_col_indices, values_values = self._compress(self.value_cache[layer_idx][:, :, :self.approximation_length], value_dictionary, self.value_max_sparsity, self.value_error_threshold)

                self._key_cache_crow_indices[layer_idx] = torch.cat([self._key_cache_crow_indices[layer_idx], keys_crow_indices[1:] + self._key_cache_crow_indices[layer_idx][-1]])
                self._key_cache_col_indices[layer_idx] = torch.cat([self._key_cache_col_indices[layer_idx], keys_col_indices])
                self._key_cache_values[layer_idx] = torch.cat([self._key_cache_values[layer_idx].to(torch.float16), keys_values.to(torch.float16)]).to(torch.float8_e4m3fn)
                self._value_cache_crow_indices[layer_idx] = torch.cat([self._value_cache_crow_indices[layer_idx], values_crow_indices[1:] + self._value_cache_crow_indices[layer_idx][-1]])
                self._value_cache_col_indices[layer_idx] = torch.cat([self._value_cache_col_indices[layer_idx], values_col_indices])
                self._value_cache_values[layer_idx] = torch.cat([self._value_cache_values[layer_idx].to(torch.float16), values_values.to(torch.float16)]).to(torch.float8_e4m3fn)

                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx][:, :, self.approximation_length:], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx][:, :, self.approximation_length:], value_states], dim=-2)
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            return compressed_key_states_to_return, keys_to_return, compressed_value_states_to_return, values_to_return
        
    def get_seq_length(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self._seen_tokens
    
    def _compress(self, tensor, dictionary, max_sparsity, error_threshold):
        CHUNK_SIZE = 20000

        head_dim = tensor.shape[-1]
        reshaped_tensor = tensor.permute(2, 0, 1, 3).reshape(1, -1, head_dim)
        
        if reshaped_tensor.shape[1] <= CHUNK_SIZE:
            indptr, indices, data = omp(
                dictionary.unsqueeze(0),
                reshaped_tensor,
                max_sparsity,
                error_threshold
            )
        else:
            chunks = reshaped_tensor.split(CHUNK_SIZE, dim=1)

            indptr = torch.tensor([0], dtype=torch.int32, device=tensor.device)
            indices = torch.empty(0, device=tensor.device)
            data = torch.empty(0, device=tensor.device)

            for chunk in chunks:
                indptr_chunk, indices_chunk, data_chunk = omp(
                    dictionary.unsqueeze(0),
                    chunk,
                    max_sparsity,
                    error_threshold
                )

                indptr = torch.cat([indptr, indptr_chunk[1:] + indptr[-1]])
                indices = torch.cat([indices, indices_chunk])
                data = torch.cat([data.to(torch.float16), data_chunk.to(torch.float16)]).to(torch.float8_e4m3fn)

        return indptr, indices, data