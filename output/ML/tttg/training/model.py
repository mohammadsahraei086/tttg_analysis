"""Neural-network architecture used as signal-vs-background classifier."""
from torch import nn

# class FeatureTokenizer(nn.Module):
#     """Turns a (batch, num_features) tensor into (batch, num_features, d_model) tokens."""
#     def __init__(self, num_features, d_model):
#         super().__init__()
#         self.num_features = num_features
#         self.d_model = d_model
#         # one weight vector + bias per feature
#         self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
#         self.bias = nn.Parameter(torch.zeros(num_features, d_model))

#     def forward(self, x):
#         # x: (batch, num_features)
#         # -> (batch, num_features, 1) * (num_features, d_model) broadcast -> (batch, num_features, d_model)
#         tokens = x.unsqueeze(-1) * self.weight + self.bias
#         return tokens

# class CLSToken(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

#     def forward(self, x):
#         # x: (batch, seq_len, d_model)
#         batch_size = x.size(0)
#         cls_tokens = self.cls.expand(batch_size, -1, -1)  # (batch, 1, d_model)
#         return torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)
        

# class OptimizedHEPClassifier(nn.Module):
#     def __init__(self, num_features=39, d_model=64, nhead=8, num_layers=3, dim_feedforward=128, dropout=0.3):
#         super().__init__()

#         self.tokenizer = FeatureTokenizer(num_features, d_model)
#         self.cls_token = CLSToken(d_model)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation='gelu',
#             batch_first=True   # critical, as discussed
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 64),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         # x: (batch, num_features)
#         tokens = self.tokenizer(x)              # (batch, num_features, d_model)
#         tokens = self.cls_token(tokens)          # (batch, num_features+1, d_model)
#         encoded = self.transformer(tokens)       # (batch, num_features+1, d_model)
#         cls_output = encoded[:, 0, :]            # take the CLS token's output
#         return self.classifier(cls_output)


class OptimizedHEPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

    # def __init__(self, input_dim):
    #     super().__init__()
    #     self.network = nn.Sequential(
    #         nn.Linear(input_dim, 64),
    #         nn.ReLU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(32, 1),
    #         # nn.Sigmoid()  # For binary classification
    #     )
    
    # def forward(self, x):
    #     return self.network(x)
