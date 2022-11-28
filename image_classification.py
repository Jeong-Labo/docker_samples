from torch.cuda import device_count
from flash import Trainer
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

def main():
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

    datamodule = ImageClassificationData.from_folders(
        train_folder="./data/hymenoptera_data/train/",
        val_folder="./data/hymenoptera_data/val/",
        test_folder="./data/hymenoptera_data/test/",
        batch_size=16
    )

    model = ImageClassifier(backbone="resnet34", labels=datamodule.labels)

    trainer = Trainer(max_epochs=5, gpus=device_count())
    trainer.finetune(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.save_checkpoint("image_classification.pt")


if __name__ == "__main__":
    main()
