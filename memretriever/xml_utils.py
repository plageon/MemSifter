from bs4 import Comment

import numpy as np
from anytree.exporter import DotExporter
from bs4 import Comment


def trim_path(path):
    #  is leaf, remove the tag
    if path["is_leaf"]:
        path["tag"].decompose()
        return
    #  not leaf, remove the text directly under the tag
    else:
        for c in path["tag"].contents:
            if not isinstance(c, bs4.element.Tag):
                # print(c)
                #  remove the text node
                c.extract()

def truncate_input(html, chat_tokenizer, max_context_window=30000):
    if isinstance(html, list):
        html = " ".join(html)
    #  if html is longer than 30000 tokens, truncate it
    tokens = chat_tokenizer.tokenize(html)
    if len(tokens) > max_context_window:
        html = chat_tokenizer.convert_tokens_to_string(tokens[:max_context_window])
        # print(f"html truncated to {max_context_window} tokens")
    return html


def simplify_html(soup, keep_attr=False):
    for script in soup(["script", "style"]):
        script.decompose()
    #  remove all attributes
    if not keep_attr:
        for tag in soup.find_all(True):
            tag.attrs = {}
    #  remove empty tags recursively
    while True:
        removed = False
        for tag in soup.find_all():
            if not tag.text.strip():
                tag.decompose()
                removed = True
        if not removed:
            break
    #  remove href attributes
    for tag in soup.find_all("a"):
        del tag["href"]
    #  remove comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    def concat_text(text):
        text = "".join(text.split("\n"))
        text = "".join(text.split("\t"))
        text = "".join(text.split(" "))
        return text

    # remove all tags with no text
    for tag in soup.find_all():
        children = [child for child in tag.contents if not isinstance(child, str)]
        if len(children) == 1:
            tag_text = tag.get_text()
            child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
            if concat_text(child_text) == concat_text(tag_text):
                tag.replace_with_children()
    #  if html is not wrapped in a html tag, wrap it

    # remove empty lines
    res = str(soup)
    lines = [line for line in res.split("\n") if line.strip()]
    res = "\n".join(lines)
    return res


class TokenIdNode(Node):
    def __init__(self, name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.input_ids = kwargs.get('input_ids', [])
        self.prob = kwargs.get('prob', np.float32(0.0))


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


