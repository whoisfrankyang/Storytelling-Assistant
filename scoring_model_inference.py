def score_pitch(abstract, generated_pitch):
    import torch
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn
    import numpy as np

    # Load tokenizer and encoders
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer_abstract = AutoModel.from_pretrained(model_name)  # X_text
    transformer_pitch = AutoModel.from_pretrained(model_name)     # X_decoded

    # Define model
    class CombinedModel(nn.Module):
        def __init__(self, transformer_abstract, transformer_pitch, mlp_input_dim, mlp_output_dim, use_mean_pooling=False):
            super().__init__()
            self.transformer_abstract = transformer_abstract
            self.transformer_pitch = transformer_pitch
            self.use_mean_pooling = use_mean_pooling

            for param in self.transformer_abstract.parameters():
                param.requires_grad = False
            for param in self.transformer_pitch.parameters():
                param.requires_grad = False

            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, mlp_output_dim)
            )

        def mean_pool(self, hidden_state, attention_mask):
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            return (hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
            outputs_1 = self.transformer_abstract(input_ids=input_ids_1, attention_mask=attention_mask_1)
            outputs_2 = self.transformer_pitch(input_ids=input_ids_2, attention_mask=attention_mask_2)

            if self.use_mean_pooling:
                emb_1 = self.mean_pool(outputs_1.last_hidden_state, attention_mask_1)
                emb_2 = self.mean_pool(outputs_2.last_hidden_state, attention_mask_2)
            else:
                emb_1 = outputs_1.last_hidden_state[:, 0, :]
                emb_2 = outputs_2.last_hidden_state[:, 0, :]

            combined = torch.cat((emb_1, emb_2), dim=1)
            return self.mlp(combined)

    # Tokenization helper
    def tokenize_pair(decoded_text, original_text, tokenizer, max_length=128):
        encoded_decoded = tokenizer(
            decoded_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoded_text = tokenizer(
            original_text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return (
            encoded_decoded['input_ids'],
            encoded_decoded['attention_mask'],
            encoded_text['input_ids'],
            encoded_text['attention_mask']
        )

    # Rebuild model
    mlp_input_dim = transformer_abstract.config.hidden_size * 2
    mlp_output_dim = 4
    model = CombinedModel(
        transformer_abstract=transformer_abstract,
        transformer_pitch=transformer_pitch,
        mlp_input_dim=mlp_input_dim,
        mlp_output_dim=mlp_output_dim,
        use_mean_pooling=False
    )

    # Load checkpoint
    model.load_state_dict(torch.load("scoring_model.pt", map_location=torch.device("cpu")))
    model.eval()

    # Tokenize and prepare
    ids1, mask1, ids2, mask2 = tokenize_pair(generated_pitch, abstract, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ids1, mask1, ids2, mask2 = ids1.to(device), mask1.to(device), ids2.to(device), mask2.to(device)

    # Inference
    with torch.no_grad():
        scores = model(ids1, mask1, ids2, mask2).cpu().numpy()[0]

    categories = ["coherence", "consistency", "fluency", "relevance"]
    return {k: float(v) for k, v in zip(categories, scores)}


abstract_path = "evaluation/benchmark_set/bandit.txt"
generated_pitch_path = "evaluation/generated_pitch/conference/bandit_conference.txt"

with open(abstract_path, "r") as f:
    abstract = f.read().strip()
with open(generated_pitch_path, "r") as f:
    pitch = f.read().strip()

scores = score_pitch(abstract, pitch)
print(scores)
