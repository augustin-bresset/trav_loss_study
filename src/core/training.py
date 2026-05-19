import numpy as np
import yaml
import logging
import os

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchsparse import SparseTensor
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import (  # TODO @abhaydmathur : Add more schedulers, move all schedulers to callbacks somehow?
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    StepLR,
)
from functools import partial

# Local Imports

from src.utils import collate_function

from src.datasets import (
    Rellis3D,
    Goose3D,
    Outback,
)

from src.losses import ClassificationCriterion
from src.utils import Logger

from .metrics import (
    confusion_matrix_from_arrays,
    stats_accuracy_per_class,
    stats_f1score_per_class,
    stats_iou_per_class,
)

from src.transforms import TorchSparseQuantize

from src.models.torchsparse_minkunet import TorchSparseMinkUNet
from src.models.torchsparse_sep import MinkUNetWithAuxDecoder  # Import the new model


DATASETS = {
    "rellis3d": Rellis3D,
    "goose": Goose3D,
    "outback": Outback,
}


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device

        self.lambda_ = float(self.args.lambda_aux)

        source_ds = list(self.args.source_datasets.keys())[0]
        self.source_dataset = DATASETS[source_ds](
            root_dir=self.args.source_datasets[source_ds]["root"],
            remap_cfg=self.args.source_datasets[source_ds].get("cfg"),
            split="train",
            transform=TorchSparseQuantize(voxel_size=self.args.voxel_size),
        )

        self.target_datasets_list = [
            DATASETS[ds](
                root_dir=self.args.target_datasets[ds]["root"],
                remap_cfg=self.args.target_datasets[ds]["cfg"],
                split="train",
                transform=TorchSparseQuantize(voxel_size=self.args.voxel_size),
            )
            for ds in self.args.target_datasets.keys()
        ]

        self.target_dataset = (
            ConcatDataset(self.target_datasets_list)
            if self.target_datasets_list
            else None
        )

        self.source_val_dataset = DATASETS[source_ds](
            root_dir=self.args.source_datasets[source_ds]["root"],
            remap_cfg=self.args.source_datasets[source_ds]["cfg"],
            split="val",
            transform=TorchSparseQuantize(voxel_size=self.args.voxel_size),
        )

        self.target_val_datasets_list = [
            DATASETS[ds](
                root_dir=self.args.target_datasets[ds]["root"],
                remap_cfg=self.args.target_datasets[ds]["cfg"],
                split="val",
                transform=TorchSparseQuantize(voxel_size=self.args.voxel_size),
            )
            for ds in self.args.target_datasets.keys()
        ]

        self.cat_item_list = [
            "pos",
            "x",
            "y",
        ]  # , "dirs", "pos_non_manifold", "occupancies"]

        if hasattr(self.args, "use_heights") and self.args.use_heights:
            self.cat_item_list.append("height_labels")

        self.stack_item_list = []
        self.sparse_item_list = ["sparse_input"]

        self.string_append_list = ["pcd_file", "label_file"]

        self.collate_fn = partial(
            collate_function,
            cat_item_list=self.cat_item_list,
            stack_item_list=self.stack_item_list,
            sparse_item_list=self.sparse_item_list,
            string_append_list=self.string_append_list,
        )

        self.source_loader = DataLoader(
            self.source_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        self.target_loader = (
            DataLoader(
                self.target_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )
            if self.target_dataset
            else None
        )

        self.source_val_loader = DataLoader(
            self.source_val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

        self.target_val_loaders = (
            {
                ds: DataLoader(
                    target_val_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn,
                )
                for ds, target_val_dataset in zip(
                    list(self.args.target_datasets.keys()),
                    self.target_val_datasets_list,
                )
            }
            if self.target_val_datasets_list
            else {}
        )

        if self.args.predictions_only:
            self.pred_datasets_list = [
                DATASETS[ds](
                    root_dir=self.args.pred_datasets[ds]["root"],
                    remap_cfg=self.args.pred_datasets[ds]["cfg"],
                    split=self.args.pred_datasets[ds]["split"],
                    transform=TorchSparseQuantize(voxel_size=self.args.voxel_size),
                )
                for ds in self.args.pred_datasets.keys()
            ]

            self.pred_loaders = {
                ds: DataLoader(
                    pred_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn,
                )
                for ds, pred_dataset in zip(
                    list(self.args.pred_datasets.keys()),
                    self.pred_datasets_list,
                )
            }

            self.pred_models = {
                model_name: MinkUNetWithAuxDecoder(
                    in_features=4,
                    num_classes=self.source_dataset.num_classes_,
                    num_auxiliary_classes=self.source_dataset.num_auxiliary_classes_,
                    voxel_size=self.args.voxel_size,
                    cylindrical_coordinates=self.args.cylindrical_coordinates,
                    cr=self.args.cr,
                    use_spatial_context=self.args.pred_models[model_name]["use_spatial_context"],
                    spatial_context_out_channels=self.args.spatial_context_out_channels,
                )
                for model_name in self.args.pred_models.keys()
            }

            print(self.pred_models)

            for model_name, model_cfg in self.args.pred_models.items():
                self.pred_models[model_name].load_state_dict(torch.load(model_cfg['ckpt']))

        else:
            self.model = MinkUNetWithAuxDecoder(
                in_features=4,
                num_classes=self.source_dataset.num_classes_,
                num_auxiliary_classes=self.source_dataset.num_auxiliary_classes_,
                voxel_size=self.args.voxel_size,
                cylindrical_coordinates=self.args.cylindrical_coordinates,
                cr=self.args.cr,
                use_spatial_context=self.args.use_spatial_context,
                spatial_context_out_channels=self.args.spatial_context_out_channels,
            )

            if hasattr(self.args, "load_ckpt") and self.args.load_ckpt is not None:
                self.model.load_state_dict(torch.load(self.args.load_ckpt))

            self.model.to_(self.device)
            print(f"Model is on {self.model.device}")

            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.opt_weight_decay,
            )

            self.scaler = torch.cuda.amp.GradScaler()

            if self.args.lr_scheduler == "plateau":
                self.lr_scheduler = ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=3
                )
            elif self.args.lr_scheduler == "cosine":
                self.lr_scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=max(len(self.source_loader) // 30, 1),
                    T_mult=1,
                    eta_min=0,
                )
            elif self.args.lr_scheduler == "step":
                self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
            else:
                raise NotImplementedError(
                    f"LR scheduler {args.lr_scheduler} not implemented"
                )

        with open(self.args.labels_cfg, "r") as file:
            self.semantic_map = yaml.safe_load(file)["semantic_map"]

        

        self.log_dir = os.path.join(self.args.log_dir, self.args.exp_name)
        self.logger = Logger(self.log_dir)

        self.segmentation_loss = ClassificationCriterion(
            num_classes=self.source_dataset.num_classes_,
            ignore_index=0,
            losses=self.args.segmentation_losses,
        )

        self.auxiliary_loss = ClassificationCriterion(
            num_classes=self.source_dataset.num_auxiliary_classes_,
            ignore_index=0,
            losses=self.args.auxiliary_losses,
        )

        self.model_name = self.args.model_name
        self.model_save_path = os.path.join(
            self.args.model_save_path, self.model_name, self.args.exp_name
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        self.training_history = {}

        self.best_model_path = None
        self.latest_model_path = None
        self.best_loss = np.inf
        self.best_acc = 0
        self.best_iou = 0
        self.best_epoch = 0

        print(f"Training on device: {self.device}")
        print(f"Saving logs to {args.log_dir}")
        print(f"Saving models to {self.model_save_path}")
        print(f"Loss functions: {', '.join(args.segmentation_losses)}")
        print(f"Training Samples : {len(self.source_dataset)}")
        print(f"Validation Samples : {len(self.source_val_dataset)}")

    def log_init(self):
        pass

    def train(self):
        self.log_init()
        for epoch in range(self.args.epochs):
            train_info = self.train_epoch(epoch)
            log_epoch_info = {f"train/{k}": v for k, v in train_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch + 1)

            val_info = self.validate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch + 1)

            self.training_history[epoch + 1] = {
                "train": train_info,
                "val": val_info,
            }

            if self.best_model_path is None or self.is_best_model(val_info):
                self.best_model_path = os.path.join(
                    self.model_save_path, "best_model.pth"
                )
                self.save(self.best_model_path)
                self.best_epoch = epoch

            if self.latest_model_path is not None:
                try:
                    os.remove(self.latest_model_path)
                except Exception as e:
                    print(e)
            self.latest_model_path = os.path.join(
                self.model_save_path, f"latest_model_{epoch+1}eps.pth"
            )
            self.save(self.latest_model_path)

            if (epoch) % self.args.test_freq == 0:
                test_metrics = self.test()
                log_test_info = {f"test/{k}": v for k, v in test_metrics.items()}
                self.save_to_log(
                    self.args.log_dir, self.logger, log_test_info, epoch + 1
                )

            if self.execute_callbacks(epoch):
                break

        test_metrics = self.test()
        log_test_info = {f"test/{k}": v for k, v in test_metrics.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_test_info, epoch + 1)

    def execute_callbacks(self, epoch):
        # Early Stopping
        if self.args.early_stopping_patience is not None:
            if epoch - self.best_epoch > self.args.early_stopping_patience:
                print(f"Early Stopping after {epoch} epochs")
                return True
        return False

    def is_best_model(self, val_info):
        if self.args.early_stopping_metric == "loss":
            k = val_info["loss"] < self.best_loss
            if k:
                self.best_loss = val_info["loss"]
            return k
        elif self.args.early_stopping_metric == "acc":
            k = val_info["acc"] > self.best_acc
            if k:
                self.best_acc = val_info["acc"]
            return k
        elif self.args.early_stopping_metric == "iou":
            k = val_info["iou"] > self.best_iou
            if k:
                self.best_iou = val_info["iou"]
            return k
        else:
            raise NotImplementedError(
                f"Early stopping withe {self.args.early_stopping_metric} not implemented"
            )

    def train_epoch(self, epoch):
        self.model.train()

        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader) if self.target_loader else None

        num_steps = len(self.source_loader)
        if self.args.max_steps is not None:
            num_steps = min(num_steps, self.args.max_steps)

        source_losses = []
        source_aux_losses = []
        target_aux_losses = []

        main_loss_infos = []

        for i in range(num_steps):

            source_batch = next(source_iter)
            source_batch = self.dict_to_device(source_batch, self.device)
            self.optimizer.zero_grad()

            aux_activate = epoch > self.args.aux_after_epoch
            with torch.amp.autocast('cuda')::
                out = self.model(source_batch, get_main=True, get_aux=aux_activate)
                
                main_output = out["main_output"]

                main_loss, main_loss_info = self.segmentation_loss(
                    main_output, source_batch["y"]
                )

                loss = main_loss
                if aux_activate:
                    aux_output = out["aux_output"]
                    aux_loss, aux_loss_info = self.auxiliary_loss(
                        aux_output, source_batch["height_labels"]
                    )
                    loss += self.lambda_ * aux_loss
                    source_aux_losses.append(aux_loss.item())
                
            # Scaler Backward
            self.scaler.scale(loss).backward()

            if hasattr(self.args, "clip_grad_norm") and self.args.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.args.clip_grad_norm)
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            main_loss_infos.append(main_loss_info)
            source_losses.append(main_loss.item())

            if self.args.lr_scheduler == "cosine":
                self.lr_scheduler.step(epoch + i / num_steps)

            # Cleaning
            del source_batch, out, main_output, main_loss, loss
            if aux_activate:
                del aux_output, aux_loss

            torch.cuda.empty_cache()

            

            if aux_activate and target_iter is not None:
                target_batch = next(target_iter)
                target_batch = self.dict_to_device(target_batch, self.device)
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    out = self.model(target_batch, get_main=False, get_aux=aux_activate)
                    aux_output = out["aux_output"]
                    aux_loss, target_aux_loss_info = self.auxiliary_loss(
                        aux_output, target_batch["height_labels"]
                    )
                self.scaler.scale(aux_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                target_aux_losses.append(aux_loss.item())

                del target_batch, out, aux_output, aux_loss
                torch.cuda.empty_cache()

            print(
                f"\rEpoch {epoch+1} [{i+1}/{num_steps}] - source_loss : {np.mean(source_losses):.3f}, source_aux_loss : {np.mean(source_aux_losses):.3f}   ",
                end="",
            )
        print()

        info = {
            "source_loss": np.mean(source_losses),
            "source_aux_loss": np.mean(source_aux_losses),
            "target_aux_loss": np.mean(target_aux_losses) if target_aux_losses else 0,
        }

        for key in main_loss_infos[0].keys():
            info[f"source_loss/{key}"] = np.mean([x[key] for x in main_loss_infos])

        return info

    def validate(self):
        self.model.eval()
        total_loss = 0
        preds = []
        labels = []
        with torch.no_grad():
            for i, batch in enumerate(self.source_val_loader):
                batch = self.dict_to_device(batch, self.device)
                out = self.model(batch, get_main=True, get_aux=False)
                main_output = out["main_output"]

                preds.extend(torch.argmax(main_output, dim=1).cpu().numpy())
                labels.extend(batch["y"].cpu().numpy())

                loss, loss_info = self.segmentation_loss(main_output, batch["y"])
                total_loss += loss.item()

                print(
                    f"\rValidation [{i+1}/{len(self.source_val_loader)}] - loss : {total_loss/(i+1):.3f}",  # , end="",
                    end="",
                )

                if self.args.max_steps is not None and i >= self.args.max_steps:
                    break
            print()

        avg_loss = total_loss / len(self.source_val_loader)
        conf_matrix = confusion_matrix_from_arrays(labels, preds)
        acc, acc_per_class = stats_accuracy_per_class(conf_matrix, ignore_list=[0])
        iou, iou_per_class = stats_iou_per_class(conf_matrix, ignore_list=[0])
        f1, f1score_per_class = stats_f1score_per_class(conf_matrix, ignore_list=[0])

        info = {
            "loss": avg_loss,
            "acc": acc,
            "iou": iou,
            "f1score": f1,
        }

        for i in range(len(acc_per_class)):
            if i not in self.semantic_map:
                self.semantic_map[i] = self.semantic_map[0]  # void
            info[f"acc/{self.semantic_map[i]}"] = acc_per_class[i]
            info[f"iou/{self.semantic_map[i]}"] = iou_per_class[i]
            info[f"f1score/{self.semantic_map[i]}"] = f1score_per_class[i]

        return info

    def dict_to_device(self, data, device):
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device)
            elif isinstance(value, list):
                data[key] = self.list_to_device(value, device)
            elif isinstance(value, dict):
                data[key] = self.dict_to_device(value, device)
            elif isinstance(value, SparseTensor):
                data[key] = data[key].to(device)
        return data

    def list_to_device(self, data, device):
        for key, value in enumerate(data):
            if torch.is_tensor(value):
                data[key] = value.to(device)
            elif isinstance(value, list):
                data[key] = self.list_to_device(value, device)
            elif isinstance(value, dict):
                data[key] = self.dict_to_device(value, device)
        return data

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def save_to_log(self, logdir, logger, info, epoch, w_summary=False, model=None):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                try:
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                except:
                    continue
                    logger.histo_summary(tag, value.data, epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + "/grad", value.grad.data.cpu().numpy(), epoch
                    )

    def test(self):
        self.model.eval()
        all_metrics = {}

        with torch.no_grad():
            for ds_name, loader in self.target_val_loaders.items():
                # all_metrics[ds_name] = {}
                all_preds = []
                all_labels = []
                for i, batch in enumerate(loader):
                    batch = self.dict_to_device(batch, self.device)
                    out = self.model(batch, get_main=True, get_aux=False)
                    main_output = out["main_output"]
                    preds = torch.argmax(main_output, dim=1).cpu().numpy()
                    labels = batch["y"].cpu().numpy()

                    all_preds.append(preds)
                    all_labels.append(labels)

                    print(
                        f"\rTesting {ds_name} [{i+1}/{len(loader)}] - ",  # , end="",
                        end="",
                    )

                    if self.args.max_steps is not None and i >= self.args.max_steps:
                        break

                print()

                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)

                conf_matrix = confusion_matrix_from_arrays(
                    all_labels, all_preds  # , self.source_dataset.num_classes_
                )
                acc, acc_per_class = stats_accuracy_per_class(
                    conf_matrix, ignore_list=[0]
                )
                iou, iou_per_class = stats_iou_per_class(conf_matrix, ignore_list=[0])
                f1, f1score_per_class = stats_f1score_per_class(
                    conf_matrix, ignore_list=[0]
                )

                all_metrics[f"{ds_name}/acc"] = acc
                all_metrics[f"{ds_name}/iou"] = iou
                all_metrics[f"{ds_name}/f1score"] = f1
                for i in range(len(acc_per_class)):
                    if i not in self.semantic_map:
                        self.semantic_map[i] = self.semantic_map[0]  # void
                    all_metrics[f"{ds_name}/acc/{self.semantic_map[i]}"] = (
                        acc_per_class[i]
                    )
                    all_metrics[f"{ds_name}/iou/{self.semantic_map[i]}"] = (
                        iou_per_class[i]
                    )
                    all_metrics[f"{ds_name}/f1score/{self.semantic_map[i]}"] = (
                        f1score_per_class[i]
                    )

        return all_metrics

    def predict(self):
        print("Predicting ...")
        with torch.no_grad():
            print(f"Models : {self.pred_models}")
            for model_name, model in self.pred_models.items():
                model.to_(self.device)
                model.eval()
                preds_dir = os.path.join(
                    os.path.dirname(self.args.pred_models[model_name]['ckpt']), "preds"
                )
                os.makedirs(preds_dir, exist_ok=True)
                print(f"Loaders : {self.pred_loaders}")
                for ds_name, loader in self.pred_loaders.items():
                    print(f"Predicting {ds_name} with {model_name} ...")
                    for i, batch in enumerate(loader):
                        print(
                            f"\rPredicting {ds_name} with {model_name} [{i+1}/{len(loader)}]", 
                            end="",
                        )
                        batch = self.dict_to_device(batch, self.device)
                        batch_ids = batch["pos"][:, 0].cpu().numpy()
                        assert np.max(batch_ids) == self.args.batch_size - 1, f"{np.max(batch_ids)} != {self.args.batch_size - 1}"
                        out = model(batch, get_main=True, get_aux=False)
                        main_output = out["main_output"]
                        preds = torch.argmax(main_output, dim=1).cpu().numpy()
                        
                        for j in range(len(batch["label_file"])):
                            pred_file = os.path.join(
                                preds_dir,
                                os.path.basename(os.path.dirname(batch["label_file"][j])),
                                os.path.basename(batch["label_file"][j]) + ".pred",
                            )

                            pred_dir = os.path.dirname(pred_file)
                            os.makedirs(pred_dir, exist_ok=True)

                            curr_pred = preds[batch_ids == j].astype(np.uint32)
                            curr_pred.tofile(pred_file)

                print()

