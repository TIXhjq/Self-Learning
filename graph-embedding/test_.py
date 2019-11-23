# _*_ coding:utf-8 _*_
<<<<<<< HEAD
from deepwalk import DeepWalk
from line import Line
from node2vec import node2vec
from sdne import sdne
from util_tool import read_graph
from evaluate import evaluate_tools

def deep_walk_run():
    Graph = read_graph('wiki/Wiki_edgelist.txt')

    deepwalk = DeepWalk(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=128,
        work=4
    )

    embeddings = deepwalk.transform()
    eval = evaluate_tools(embeddings=embeddings,label_path='wiki/Wiki_labels.txt')
    eval.plot_embeddings()

def line_run():
    from util import read_graph
    import os
    print(os.getcwd())
    Graph = read_graph('wiki/Wiki_edgelist.txt')
    line = Line(
        Graph=Graph,
        dimension_size=128,
        per_vertex=100,
        walk_length=10,
        window_size=5,
        work=1,
        negative_ratio=1,
        batch_size=128,
        log_dir='logs/0/',
        epoch=100,
    )
    embeddings = line.transform()
    from evaluate import evaluate_tools
    tool = evaluate_tools(embeddings,label_path='wiki/Wiki_labels.txt')
    tool.plot_embeddings()

def node2vec_run():
    Graph = read_graph('wiki/Wiki_edgelist.txt')

    node_vec = node2vec(
        Graph=Graph,
        per_vertex=80,
        walk_length=10,
        window_size=5,
        dimension_size=128,
        work=1,
        p=0.25,
        q=4
    )

    embeddings = node_vec.transform()
    eval_tool = evaluate_tools(embeddings,label_path='wiki/Wiki_labels.txt')
    eval_tool.plot_embeddings()

def sdne_run():
    Graph = read_graph('wiki/Wiki_edgelist.txt')
    sden_model = sdne(
        Graph=Graph,
        dimension_size=128,
        per_vertex=100,
        walk_length=10,
        window_size=5,
        work=1,
        beta=5,
        alpha=1e-6,
        verbose=1,
        epochs=1000,
        batch_size=512,
        log_dir='logs/0/',
        hidden_size_list=[256, 128],
        l1=1e-5,
        l2=1e-4
    )
    sden_model.train()
    embeddings = sden_model.get_embeddings()

    from evaluate import evaluate_tools
    eval_tool = evaluate_tools(embeddings,label_path='wiki/Wiki_labels.txt')
    eval_tool.plot_embeddings()


def test(build_name):
    if build_name=='deepwalk':
        deep_walk_run()
    elif build_name=='line':
        line_run()
    elif build_name=='node2vec':
        node2vec_run()
    elif build_name=='sdne':
        sdne_run()
    elif build_name=='all':
        deep_walk_run()
        line_run()
        node2vec_run()
        sdne_run()

if __name__=='__main__':
    test('sdne')
=======
import numpy as np
print(np.ones(12)*-1)
>>>>>>> parent of d1d9e41... deepwalk,line,node2vec
