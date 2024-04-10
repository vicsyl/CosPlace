
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset

import infer_visualizations


def infer(args: Namespace, eval_ds: Dataset, model: torch.nn.Module,
         num_preds_to_save: int = 0) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        logging.debug("DB FILES")
        for images, files, indices in tqdm(database_dataloader, ncols=100):
            logging.debug(f"files: {files}")
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 5
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        logging.debug("QUERY FILES")
        for images, files, indices in tqdm(queries_dataloader, ncols=100):
            logging.debug(f"files: {files}")
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logging.debug("Calculating recalls")
    retrieved_count = 1
    distance, predictions = faiss_index.search(queries_descriptors, retrieved_count)

    logging.debug(f"distances: {distance}")
    logging.debug(f"predictions: {predictions}")

    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        infer_visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, args.output_folder, args.save_only_wrong_preds)

