from typing import Optional, Tuple, Union
from numpy import isin
import transformers
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaLayer, RobertaSelfOutput, RobertaAttention, RobertaOutput, RobertaEncoder, RobertaPooler
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from transformers_filter import FilterModel, AutoConfig
import torch
import torch.nn as nn

def surrogate_pre_forward(bert_model:Union[transformers.RobertaModel, transformers.BertModel], input_batch:transformers.BatchEncoding, fusion_layer:int):
    output_attentions = bert_model.config.output_attentions
    output_hidden_states = (
        bert_model.config.output_hidden_states
    )
    return_dict = bert_model.config.use_return_dict

    if bert_model.config.is_decoder:
        use_cache = bert_model.config.use_cache
    else:
        use_cache = False

    if fusion_layer == 0:
        embeddings = bert_model.embeddings.word_embeddings(input_batch["input_ids"])
        return embeddings
    else:
        input_ids = input_batch["input_ids"]
        input_shape = input_ids.size()

        batch_size, seq_length = input_shape
        device = input_ids.device
        # past_key_values_length
        past_key_values_length = 0

        attention_mask = input_batch['attention_mask']
        if hasattr(bert_model.embeddings, "token_type_ids"):
            buffered_token_type_ids = bert_model.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = bert_model.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_extended_attention_mask = None
        head_mask = bert_model.get_head_mask(None, bert_model.config.num_hidden_layers)

        past_key_values = None
        encoder_hidden_states = None
        attention_mask = extended_attention_mask
        encoder_attention_mask = encoder_extended_attention_mask
        
        embedding_output = bert_model.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=past_key_values_length,
        )

        hidden_states = embedding_output
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and bert_model.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(bert_model.encoder.layer):
            if i == fusion_layer:
                break
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if bert_model.encoder.gradient_checkpointing and bert_model.encoder.training:

                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            # if use_cache:
            #     next_decoder_cache += (layer_outputs[-1],)
            # if output_attentions:
            #     all_self_attentions = all_self_attentions + (layer_outputs[1],)
            #     if bert_model.config.add_cross_attention:
            #         all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states
        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [
        #             hidden_states,
        #             next_decoder_cache,
        #             all_hidden_states,
        #             all_self_attentions,
        #             all_cross_attentions,
        #         ]
        #         if v is not None
        #     )
        # return BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_decoder_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attentions,
        #     cross_attentions=all_cross_attentions,
        # )


def surrogate_post_forward(bert_model:Union[transformers.RobertaModel, transformers.BertModel], input_batch:transformers.BatchEncoding, fusion_layer:int, hidden_states:torch.FloatTensor, reference_matrix:torch.FloatTensor, reference_weights:torch.FloatTensor):
    output_attentions = bert_model.config.output_attentions
    output_hidden_states = (
        bert_model.config.output_hidden_states
    )
    return_dict = bert_model.config.use_return_dict

    if bert_model.config.is_decoder:
        use_cache = bert_model.config.use_cache
    else:
        use_cache = False

    if isinstance(reference_weights, tuple):
        hidden_states = reference_weights[0] * hidden_states + reference_weights[1] * reference_matrix
    else:
        hidden_states = reference_weights * reference_matrix + (1 - reference_weights) * hidden_states

    
    input_ids = input_batch["input_ids"]
    input_shape = input_ids.size()

    batch_size, seq_length = input_shape
    device = input_ids.device
    # past_key_values_length
    past_key_values_length = 0

    attention_mask = input_batch['attention_mask']
    if hasattr(bert_model.embeddings, "token_type_ids"):
        buffered_token_type_ids = bert_model.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
    else:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    extended_attention_mask: torch.Tensor = bert_model.get_extended_attention_mask(attention_mask, input_shape, device)
    encoder_extended_attention_mask = None
    head_mask = bert_model.get_head_mask(None, bert_model.config.num_hidden_layers)

    past_key_values = None
    encoder_hidden_states = None
    attention_mask = extended_attention_mask
    encoder_attention_mask = encoder_extended_attention_mask
    if fusion_layer == 0:
        hidden_states = bert_model.embeddings(
            input_ids=None,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )

    
    # all_hidden_states = () if output_hidden_states else None
    # all_self_attentions = () if output_attentions else None
    # all_cross_attentions = () if output_attentions and bert_model.config.add_cross_attention else None

    # next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(bert_model.encoder.layer):
        if i < fusion_layer:
            continue
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if bert_model.encoder.gradient_checkpointing and bert_model.encoder.training:

            if use_cache:
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        hidden_states = layer_outputs[0]
        # if use_cache:
        #     next_decoder_cache += (layer_outputs[-1],)
        # if output_attentions:
        #     all_self_attentions = all_self_attentions + (layer_outputs[1],)
        #     if bert_model.config.add_cross_attention:
        #         all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    # if output_hidden_states:
    #     all_hidden_states = all_hidden_states + (hidden_states,)

    return hidden_states
    # if not return_dict:
    #     return tuple(
    #         v
    #         for v in [
    #             hidden_states,
    #             next_decoder_cache,
    #             all_hidden_states,
    #             all_self_attentions,
    #             all_cross_attentions,
    #         ]
    #         if v is not None
    #     )
    # return BaseModelOutputWithPastAndCrossAttentions(
    #     last_hidden_state=hidden_states,
    #     past_key_values=next_decoder_cache,
    #     hidden_states=all_hidden_states,
    #     attentions=all_self_attentions,
    #     cross_attentions=all_cross_attentions,
    # )
