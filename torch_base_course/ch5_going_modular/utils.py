import torch

from pathlib import Path

def save_model(model: torch.nn.Module,
               save_dir: str,
               model_name: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt")

    save_path = save_dir / model_name
    print(f'model save path: {save_path}')

    torch.save(obj=model.state_dict(), f=save_path)
    print(f'Save model success !!!')

if __name__=='__main__':

    torch.manual_seed(42)
    model = torch.nn.Linear(in_features=2, out_features=2)
    print(f'model state dict: {model.state_dict()}')

    save_dir = "./models"

    model_name = "linear_layer.pth"
    save_model(model, save_dir, model_name)

    model_loaded = torch.nn.Linear(in_features=2, out_features=2)
    model_loaded.load_state_dict(torch.load("./models/linear_layer.pth"))
    print(f'loaded model state dict: {model_loaded.state_dict()}')