
import sys
from multiprocessing import freeze_support

import torch
import logging
import multiprocessing
from datetime import datetime

import infer
import parser
import commons
from cosplace_model import cosplace_network
from datasets.infer_dataset import InferDataset

if __name__ == '__main__':

    # --backbone ResNet50 --fc_output_dim 2048 --resume_model /Users/vaclav/.cache/torch/hub/checkpoints/ResNet50_2048_cosplace.pth --test_set_folder amstertime/images/test --num_preds_to_save=3 --device cpu
    # import sys
    # sys.argv.extend(["--backbone", "ResNet50",
    #                  "--fc_output_dim", "2048",
    #                  "--resume_model", "<local_path>",
    #                  "--test_set_folder", "amstertime/images/test_name/",
    #                  "--num_preds_to_save", "3",
    #                  "--device", "cpu"])
    # freeze_support()

    torch.backends.cudnn.benchmark = True  # Provides a speedup

    args = parser.parse_arguments(is_training=False)
    start_time = datetime.now()
    args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.output_folder}")

    #### Model
    model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.info(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model, torch.device("cpu"))
        model.load_state_dict(model_state_dict)
    else:
        logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                     "Evaluation will be computed using randomly initialized weights.")

    model = model.to(args.device)

    infer_ds = InferDataset(args.test_set_folder, queries_folder="queries_v1",
                            positive_dist_threshold=args.positive_dist_threshold)

    infer.infer(args, infer_ds, model, args.num_preds_to_save)
