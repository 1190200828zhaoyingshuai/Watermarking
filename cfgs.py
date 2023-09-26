class Traincfg:
    """
    训练参数设置
    """
    def __init__(self,batch_size: int,
                 number_of_epochs: int,
                 train_folder: str, valid_folder: str, save_folder: str,
                 start_epoch: int,
                 H: int, W: int, message_length: int,
                 decoder_loss: float,
                 mse_weight: float,
                 encoder_loss: float,
                 adversarial_loss: float):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.valid_folder = valid_folder
        self.save_folder = save_folder
        self.start_epoch = start_epoch
        self.H = H
        self.W = W
        self.message_length = message_length
        self.decoder_loss = decoder_loss
        self.mse_weight = mse_weight
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
