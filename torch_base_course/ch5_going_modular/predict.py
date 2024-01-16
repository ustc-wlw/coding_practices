import torch
from torchvision import transforms
import torchvision


from model_builder import TinyVGG

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_img(img_path: str,
             model_path: str=None,
             tranform: transforms=None,
             device: torch.device=torch.device("cpu")):
    img_uint8 = torchvision.io.read_image(img_path)

    print(f'img shape: {img_uint8.shape}')
    print(f'img dtype: {img_uint8.dtype}')

    img_fp32 = img_uint8.type(torch.float32) / 255
    print(f'img dtype: {img_fp32.dtype}')

    if tranform:
        img = tranform(img_fp32)
    else:
        img = img_fp32
    img = img.unsqueeze(dim=0).to(device)
    print(f'img shape: {img.shape}')
    print(f'img dtype: {img.dtype}')

    assert model_path and model_path.endswith(".pth")

    model = TinyVGG(3, 10, 3)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # print(f'model is {model}')
    model.eval()
    with torch.inference_mode():
        logits = model(img)
        prediction = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        print(f'prediction label id is : {prediction}')


if __name__=="__main__":
    test_img_path = "./04-pizza-dad.jpeg"

    transform = transforms.Compose([
        transforms.Resize((64,64)),
    ])

    model_path = "./models/tinyVGG.pth"

    test_img(test_img_path, model_path=model_path, tranform=transform)
