from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, exp_name=None):
        self.writer = SummaryWriter(exp_name)

    def write_train(self, loss, bleu, rouge, steps):
        '''
        Args:
            loss: scalar
            bleu: scalar
            rouge: [1,2,L,BE]
            steps: scalar, training steps
        '''
        self.writer.add_scalar('train/loss', loss, steps)
        self.writer.add_scalar('train/bleu', bleu, steps)
        self.writer.add_scalar('train/rouge_1', rouge[0], steps)
        self.writer.add_scalar('train/rouge_2', rouge[1], steps)
        self.writer.add_scalar('train/rouge_L', rouge[2], steps)
        self.writer.add_scalar('train/rouge_BE', rouge[3], steps)

    def write_valid(self, loss, bleu, rouge, epoch):
        '''
        Args:
            loss: scalar
            bleu: scalar
            rouge: [1,2,L,BE]
            epoch: scalar, training epochs
        '''
        self.writer.add_scalar('valid/loss', loss, epoch)
        self.writer.add_scalar('valid/bleu', bleu, epoch)
        self.writer.add_scalar('valid/rouge_1', rouge[0], epoch)
        self.writer.add_scalar('valid/rouge_2', rouge[1], epoch)
        self.writer.add_scalar('valid/rouge_L', rouge[2], epoch)
        self.writer.add_scalar('valid/rouge_BE', rouge[3], epoch)

