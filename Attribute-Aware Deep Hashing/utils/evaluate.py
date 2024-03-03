import msadapter.pytorch as torch
from tqdm import tqdm
def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0
    ndcg_all = []

    for i in tqdm(range(num_query)):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())
        index = torch.argsort(hamming_dist)

        # Arrange position according to hamming distance
        retrieval = retrieval[index[:topk]]
        #ndcg_all.append(NDCG(query_labels[i], database_labels[index[:20]]))

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    #ndcg = sum(ndcg_all)/len(ndcg_all)
    #print(f'NDCG: {ndcg}')
    return mean_AP


def NDCG(query_label, database_labels, k=20):
    nums = [20, 13, 55, 7, 28, 11, 26, 8, 17]
    rel = None
    for i in range(9):
        l,r=sum(nums[:i]),sum(nums[:i+1])
        t = query_label[l:r]@database_labels[:, l:r].t()
        if rel is None:
            rel = t
        else:
            rel += t

    res = (2**rel - 1) / torch.log(torch.arange(k, device=query_label.device) + 2)
    res = res.sum().item()/10
    return res
    