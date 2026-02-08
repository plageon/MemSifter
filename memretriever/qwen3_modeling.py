from typing import Optional, Union, List, Tuple

import bs4
import numpy as np
import torch
from anytree import Node, PreOrderIter
from anytree.exporter import DotExporter
from torch import nn
from transformers import Qwen3PreTrainedModel, Qwen3Model
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple


def nodenamefunc(node):
    return f"{node.name}|{node.prob}|{node.input_ids}"


class TokenDotExporter(DotExporter):
    def __init__(self, node, **kwargs):
        super().__init__(node, **kwargs)

    def __iter__(self):
        # prepare
        indent = " " * self.indent
        nodenamefunc = self.nodenamefunc or self._default_nodenamefunc
        nodeattrfunc = self.nodeattrfunc or self._default_nodeattrfunc
        edgeattrfunc = self.edgeattrfunc or self._default_edgeattrfunc
        edgetypefunc = self.edgetypefunc or self._default_edgetypefunc
        filter_ = self.filter_ or self._default_filter
        return self.__iter(indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc, filter_)

    def __iter_nodes(self, indent, nodenamefunc, nodeattrfunc, filter_):
        for node in PreOrderIter(self.node, filter_=filter_, stop=self.stop, maxlevel=self.maxlevel):
            nodename = nodenamefunc(node)
            nodeattr = nodeattrfunc(node)
            nodeattr = " {%s}" % nodeattr if nodeattr is not None else ""
            yield '%s%s' % (DotExporter.esc(nodename), nodeattr)

    def __iter(self, indent, nodenamefunc, nodeattrfunc, edgeattrfunc, edgetypefunc, filter_):
        for node in self.__iter_nodes(indent, nodenamefunc, nodeattrfunc, filter_):
            yield node

class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))

@auto_docstring
class Qwen3ForGenRetrieval(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @can_return_tuple
    @auto_docstring
    def gen_retrieve(
            self,
            tokenizer,
            query: str,
            xml_his: str,
            block_tree: List[Tuple],
            **kwargs):
        max_seq_length = kwargs.pop("max_seq_length", 131072)
        template = kwargs.pop("template", "")

        def apply_html_tree_template(current, xml):
            return template.format(history=xml, current=current)

        #  get the generation probability of tree nodes
        model_input = apply_html_tree_template(query, xml_his)

        inputs = tokenizer.apply_chat_template([{"role": "user", "content": model_input}], add_special_tokens=True,
                                               add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                               return_dict=True)

        #  merge htmls to a single html
        soup = bs4.BeautifulSoup(xml_his, 'html.parser')

        token_id_paths = []
        is_leaf = [p[2] for p in block_tree]
        block_tree = [p[1] for p in block_tree]

        for path in block_tree:
            path_str = "<" + "><".join(path) + ">"
            token_ids = tokenizer.encode(path_str, add_special_tokens=False)
            token_id_paths.append(token_ids)

        res_html_refs = []
        #  construct token_id_tree
        root = TokenIdNode(-1)
        for path in token_id_paths:
            parent = root
            #  iterate through path
            for i, token_id in enumerate(path):
                has_child = False
                #  find existing child
                for child in parent.children:
                    if child.name == token_id:
                        parent = child
                        has_child = True
                        break
                if not has_child:
                    node = TokenIdNode(token_id, parent=parent, input_ids=path[:i + 1])
                    parent = node

        node_queue = [root]
        while node_queue:
            cur_node = node_queue.pop(0)
            children = cur_node.children
            if len(children) == 1:
                cur_node.children[0].prob = str(np.float32(1.0))
                node_queue.append(children[0])
                continue
            elif len(children) == 0:
                continue
            #  calculate transition probability for each child
            force_token_id = [c.name for c in children]
            child_input_ids = torch.tensor(cur_node.input_ids, dtype=torch.long).unsqueeze(0)
            # concatenate context input id with child input id
            child_input_ids = torch.cat([inputs["input_ids"][0], child_input_ids], dim=1).to(self.device)
            model_inputs = self.prepare_inputs_for_generation(child_input_ids, **kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
            )
            #  get the probability of force_token_id
            force_token_id = torch.tensor(force_token_id, device=self.device)
            probs = torch.gather(outputs.logits[:, 0, :], -1, force_token_id.unsqueeze(0))
            # softmax
            probs = torch.nn.functional.softmax(probs, dim=-1)
            # . linear transformation
            # probs = probs / probs.sum()
            probs = probs.squeeze(0).detach().to(torch.float32).cpu().numpy()
            for i, child in enumerate(children):
                child.prob = str(probs[i])
                node_queue.append(child)

            res_html_refs.append({
                "html": str(soup),
                "paths": block_tree,
                "is_leaf": is_leaf,
                "path_token_ids": token_id_paths,
                "node_tree": list(TokenDotExporter(root, nodenamefunc=nodenamefunc))
            })
        return res_html_refs
