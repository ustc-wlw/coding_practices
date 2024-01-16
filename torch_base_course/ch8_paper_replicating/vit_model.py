import torch
import torchvision
import torch.nn as nn

from torchinfo import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3,
                 patch_size=16,
                 embedding_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=embedding_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, img):
        img_resolution = img.shape[-1]
        assert img_resolution % self.patch_size == 0

        x = self.conv(img)
        x = self.flatten(x).permute(0, 2, 1)
        return x

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768,
                 num_heads=12,
                 atten_dropout=0):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim,
                                         num_heads=num_heads,
                                         dropout=atten_dropout,
                                         batch_first=True)
        self.ln_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(self, hidden):
        x = self.ln_norm(hidden)
        attn_output, _ = self.mha(query=x,
                                  key=x,
                                  value=x,
                                  need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim=768,
                 mlp_size=3072,
                 dropout=0.1):
        super().__init__()

        self.ln_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, hidden):
        x = self.ln_norm(hidden)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim=768,
                 num_heads=12,
                 atten_dropout=0,
                 mlp_size=3072,
                 dropout=0.1):
        super().__init__()

        self.attention = MultiheadSelfAttentionBlock(embedding_dim, num_heads, atten_dropout)

        self.mlp = MLPBlock(embedding_dim, mlp_size, dropout)

    def forward(self, hidden):
        x = hidden
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

class ViT(nn.Module):
    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 patch_size=16,
                 n_layers=12,
                 embedding_dim=768,
                 num_heads=12,
                 atten_dropout=0,
                 mlp_size=3072,
                 mlp_dropout=0.1,
                 embedding_dropout=0.1,
                 class_num=3
                 ):
        super().__init__()

        assert img_size % patch_size == 0, "invalid image size and patch size"

        self.num_patches = img_size ** 2 // (patch_size ** 2)

        ## learnable class embedding token
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)

        ## learnable position embedding vector
        self.position_embedding = nn.Parameter(torch.randn(1, (self.num_patches+1), embedding_dim), requires_grad=True)

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.patcher = PatchEmbedding(embedding_dim=embedding_dim,
                                      in_channels=in_channels,
                                      patch_size=patch_size)

        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim, num_heads, atten_dropout, mlp_size, mlp_dropout) for _ in range(n_layers)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=class_num)
        )

    def forward(self, img):
        print(f'input img shape is {img.shape}')
        batch_size = img.shape[0]
        patches_embedding = self.patcher(img)
        print(f'patches_embedding shape {patches_embedding.shape}')

        class_token_expanded = self.class_token.expand(batch_size, -1, -1)
        print(f'class_token after expanded: {class_token_expanded.shape}')

        patch_embedding_with_class = torch.cat([class_token_expanded, patches_embedding], dim=1)
        print(f'patch_embedding_with_class shape: {patch_embedding_with_class.shape}')

        patch_embedding_with_class_position = patch_embedding_with_class + self.position_embedding
        print(f'patch_embedding_with_class_position shape is {patch_embedding_with_class_position.shape}')

        patch_embedding_with_class_position = self.embedding_dropout(patch_embedding_with_class_position)

        hidden_after_encoder = self.encoder(patch_embedding_with_class_position)
        print(f'hidden_after_encoder shape is {hidden_after_encoder.shape}')

        logits = self.classifier(hidden_after_encoder[:, 0, :])
        print(f'logits shape is {logits.shape}')
        return logits


def test_ViT():
    img = torch.randn(1, 3, 224, 224)

    vit = ViT(class_num=3)

    summary(model=vit,
            input_size=(1, 3, 224, 224),  # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

    # vit(img)
def test_TransformerEncoderBlock2():
    torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                           nhead=12,
                                                           dim_feedforward=3072,
                                                           dropout=0.1,
                                                           activation="gelu",
                                                           batch_first=True,
                                                           norm_first=True)
    print(torch_transformer_encoder_layer)

    # # Get the output of PyTorch's version of the Transformer Encoder (uncomment for full output)
    summary(model=torch_transformer_encoder_layer,
            input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
def test_TransformerEncoderBlock():
    hidden = torch.randn(32, 197, 768)
    print(f'input hidden shape: {hidden.shape}')
    transformer = TransformerEncoderBlock()
    ret = transformer(hidden)
    print(f'output tensor shape after MLPBlock is {ret.shape}')

    summary(model=transformer,
            input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])

def test_MLPBlock():
    hidden = torch.randn(32, 197, 768)
    print(f'input hidden shape: {hidden.shape}')
    mlp = MLPBlock()
    ret = mlp(hidden)
    print(f'output tensor shape after MLPBlock is {ret.shape}')

def test_MultiheadSelfAttentionBlock():
    hidden = torch.randn(32, 197, 768)
    print(f'input hidden shape: {hidden.shape}')
    mha = MultiheadSelfAttentionBlock()
    ret = mha(hidden)
    print(f'output tensor shape after MultiheadSelfAttentionBlock is {ret.shape}')


def test_PatchEmbedding():
    img = torch.randn(32, 3, 224, 224)
    patcher = PatchEmbedding()
    patches_embedding = patcher(img)
    print(f'ret shape: {patches_embedding.shape}')

    batch_size = patches_embedding.shape[0]
    hidden_dim = patches_embedding.shape[-1]

    # Create random input sizes
    random_input_image = (1, 3, 224, 224)
    random_input_image_error = (1, 3, 250, 250)  # will error because image size is incompatible with patch_size

    # # Get a summary of the input and outputs of PatchEmbedding (uncomment for full output)
    # summary(PatchEmbedding(),
    #         input_size=random_input_image_error, # try swapping this for "random_input_image_error"
    #         col_names=["input_size", "output_size", "num_params", "trainable"],
    #         col_width=20,
    #         row_settings=["var_names"])

    class_ebemdding = nn.Parameter(torch.randn(batch_size, 1, hidden_dim), requires_grad=True)
    print(f'class embedding shape: {class_ebemdding.shape}')
    patch_embedding_with_class = torch.cat([class_ebemdding, patches_embedding], dim=1)
    print(f'patch_embedding_with_class shape: {patch_embedding_with_class.shape}')

    sequence_len = patch_embedding_with_class.shape[1]
    position_embedding = nn.Parameter(torch.randn(1, sequence_len, hidden_dim), requires_grad=True)
    print(f'position_embedding shape: {position_embedding.shape}')

    patch_embedding_with_class_position = patch_embedding_with_class + position_embedding
    print(f'patch_embedding_with_class_position shape is {patch_embedding_with_class_position.shape}')


def main():
    # test_PatchEmbedding()
    # test_MultiheadSelfAttentionBlock()
    # test_MLPBlock()
    # test_TransformerEncoderBlock2()
    test_ViT()

if __name__ == "__main__":
    main()