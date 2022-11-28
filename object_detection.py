from torch.cuda import device_count
from flash import Trainer
from flash.core.data.utils import download_data
from flash.image import ObjectDetectionData, ObjectDetector

def main():
    download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

    datamodule = ObjectDetectionData.from_coco(
        train_folder="data/coco128/images/train2017/",
        train_ann_file="data/coco128/annotations/instances_train2017.json",
        val_split=0.1,
        transform_kwargs={"image_size": 512},
        batch_size=4,
    )

    model = ObjectDetector(head="efficientdet", backbone="d0", num_classes=datamodule.num_classes, image_size=512)

    trainer = Trainer(max_epochs=5, gpus=device_count())
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")
    trainer.save_checkpoint("object_detection_model.pt")


if __name__ == "__main__":
    main()