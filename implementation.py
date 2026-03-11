import numpy as np
import pandas as pd

d_model = 64
d_ff = 256
N = 6
epsilon = 1e-6

vocab_dict = {"o": 0, "banco": 1, "bloqueou": 2, "meu": 3, "cartao": 4}
df_vocab = pd.DataFrame(list(vocab_dict.items()), columns=['Palavra', 'ID'])

frase = "o banco bloqueou meu cartao"
lista_ids = [vocab_dict[p] for p in frase.split()]

vocab_size = len(vocab_dict)
embeddings_table = np.random.randn(vocab_size, d_model)

sequencia_vetores = np.array([embeddings_table[id_word] for id_word in lista_ids])
X = np.expand_dims(sequencia_vetores, axis=0)

def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class EncoderLayer:
    def __init__(self, d_model, d_ff):
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def scaled_dot_product_attention(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        d_k = K.shape[-1]
        K_T = K.transpose(0, 2, 1)
        
        scores = (Q @ K_T) / np.sqrt(d_k)
        attention_weights = softmax(scores)
        return attention_weights @ V

    def layer_norm(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + epsilon)

    def feed_forward(self, x):
        linear_1 = (x @ self.W1) + self.b1
        ativacao_relu = np.maximum(0, linear_1)
        return (ativacao_relu @ self.W2) + self.b2

    def forward(self, x):
        x_att = self.scaled_dot_product_attention(x)
        x_norm1 = self.layer_norm(x + x_att)
        x_ffn = self.feed_forward(x_norm1)
        x_out = self.layer_norm(x_norm1 + x_ffn)
        return x_out

camadas_encoder = [EncoderLayer(d_model, d_ff) for _ in range(N)]
Z = X

print(f"Entrada (Batch, Tokens, Dim): {Z.shape}")

for camada in camadas_encoder:
    Z = camada.forward(Z)

print(f"Saida (Batch, Tokens, Dim): {Z.shape}")
