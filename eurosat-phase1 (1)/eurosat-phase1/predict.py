import argparse
from PIL import Image
import torch

from data import make_transform
from model import load_model
from utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--checkpoint', default='outputs/run1/best_model.pt')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    if args.top_k < 1:
        p.error('top-k must be positive.')
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    with Image.open(args.image) as image:
        tensor = make_transform(checkpoint['image_size'])(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(args.top_k, len(checkpoint['classes'])))
    for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
        print(f'{checkpoint["classes"][index]:24s} {score:.2%}')
    print('Scores are softmax outputs, not calibrated probabilities of correctness.')


if __name__ == '__main__':
    main()
