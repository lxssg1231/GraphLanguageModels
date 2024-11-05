from models.graph_T5.classifier import GraphT5Classifier
from models.graph_T5.wrapper_functions import Graph, graph_to_graphT5, get_embedding, Data, add_text_to_graph_data
import torch
from typing import List

def get_batch(data_instances: List[Data], pad_token_id: int, device: str):
    """
    获取批次数据的简化版本，类似于 experiments/encoder/train_LM.py 中的 get_batch 函数。

    参数:
    - data_instances (List[Data]): 数据实例列表，每个实例包含输入 ID、相对位置矩阵、稀疏掩码等。
    - pad_token_id (int): 填充标记的 ID。
    - device (str): 设备类型，例如 'cpu' 或 'cuda'。

    返回:
    - input_ids (torch.Tensor): 输入 ID 的张量。
    - relative_position (torch.Tensor): 相对位置矩阵的张量。
    - sparsity_mask (torch.Tensor): 稀疏掩码的张量。
    - use_additional_bucket (torch.Tensor): 是否使用额外桶的张量。
    - indices (List[List[int]]): 每个数据实例的索引列表。
    """
    # 确定批次中的最大序列长度
    max_seq_len = max([data.input_ids.shape[1] for data in data_instances])

    # 初始化张量
    input_ids = torch.ones((len(data_instances), max_seq_len), dtype=torch.long, device=device) * pad_token_id
    relative_position = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.long, device=device)
    sparsity_mask = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)
    use_additional_bucket = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)

    # 填充张量
    for i, data in enumerate(data_instances):
        input_ids[i, :data.input_ids.shape[1]] = data.input_ids
        relative_position[i, :data.relative_position.shape[1], :data.relative_position.shape[2]] = data.relative_position
        sparsity_mask[i, :data.sparsity_mask.shape[1], :data.sparsity_mask.shape[2]] = data.sparsity_mask
        use_additional_bucket[i, :data.use_additional_bucket.shape[1], :data.use_additional_bucket.shape[2]] = data.use_additional_bucket

    indices = [data.indices for data in data_instances]

    return input_ids, relative_position, sparsity_mask, use_additional_bucket, indices

def main():
    # 定义随机参数
    num_classes = 5
    modelsize = "t5-small"
    init_additional_buckets_from = 1e6

    # 定义测试输入（2 个实例以实现批处理）
    graph1 = [
        ("dog", "is a", "animal"),
        ("cat", "is a", "animal"),
        ("black poodle", "is a", "dog"),
    ]
    graph2 = [
        ("subject1", "relation1", "object1"),
        ("subject2", "relation2", "object1"),
        ("subject3", "relation3", "subject1"),  # subject1 是这个三元组的对象
    ]
    graphs = [Graph(graph1), Graph(graph2)]
    query_concepts = ["dog", "relation1"]  # 要分类的概念或关系。可以是关系，只要该关系只出现一次。例如，这可以用于预测被遮盖的关系。
    texts = [
        "The black poodle chases a cat.", 
        "This is an example text for the second graph."
    ]

    # 4 种不同的分类器（lGLM 和 gGLM，有无文本）
    params = [
        {
            'name': 'lGLM w/o text', 
            'num_additional_buckets': 0, 
            'how': 'local', 
            'use_text': False
        }, 
        {   
            'name': 'gGLM w/o text', 
            'num_additional_buckets': 1,  # gGLM 需要一个额外的桶来表示全局图到图的相对位置
            'how': 'global',
            'use_text': False
        },
        {
            'name': 'lGLM w/ text',
            'num_additional_buckets': 2,  # 文本到图和图到文本各需要一个额外的桶
            'how': 'local',
            'use_text': "FullyConnected"
        },
        {
            'name': 'gGLM w/ text',
            'num_additional_buckets': 3,  # 文本到图、图到文本和图到图各需要一个额外的桶
            'how': 'global',
            'use_text': "FullyConnected"
        }
    ]

    for param in params:
        # 加载模型
        model = GraphT5Classifier(config=GraphT5Classifier.get_config(num_classes=num_classes, modelsize=modelsize, num_additional_buckets=param["num_additional_buckets"]))

        # 初始化额外的桶。额外的桶是在 gGLM 和文本引导的模型中引入的额外相对位置。
        if param["num_additional_buckets"] > 0:
            model.t5model.init_relative_position_bias(modelsize=modelsize, init_decoder=False, init_additional_buckets_from=init_additional_buckets_from)

        print()  # 加载模型时会有一些警告，因为解码器参数未使用，并且如果使用了额外的桶，还会有一个警告提示这些桶是随机初始化的。`init_relative_position_bias` 内部也会加载模型，因此警告会打印两次。
        print(f"Classifier: {param['name']}")

        # 预处理数据，即将图（和可选的文本）转换为相对位置矩阵、稀疏矩阵、输入 ID 等
        data = []
        for g, t in zip(graphs, texts):
            tmp_data = graph_to_graphT5(g, model.tokenizer, how=param["how"], eos=False)
            add_text_to_graph_data(data=tmp_data, text=t, tokenizer=model.tokenizer, use_text=param["use_text"])
            data.append(tmp_data)

        # 获取批次
        input_ids, relative_position, sparsity_mask, use_additional_bucket, indices = get_batch(data, model.tokenizer.pad_token_id, "cpu")

        # 前向传播
        logits = model.forward(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )
        print(f'{logits.shape = } (batch_size, max_seq_len, num_classes)')

        # 获取查询概念的 logits
        query_logits = torch.cat([
            get_embedding(sequence_embedding=logits[i], indices=indices[i], concept=query_concepts[i], embedding_aggregation='mean')
            for i in range(len(data))
        ], dim=0)
        print(f'{query_logits.shape = } (batch_size, num_classes)')
        print(f'predicted classes: {query_logits.argmax(dim=1)} (pred_1, pred_2, ..., pred_batch_size)')
        print()

if __name__ == '__main__':
    main()
