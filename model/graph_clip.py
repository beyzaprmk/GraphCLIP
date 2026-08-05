# Ana PyTorch nn.Module sınıfı

import torch
import torch.nn as nn
# PyTorch Geometric (PyG) kütüphanesinden GNN katmanları
from torch_geometric.nn import GATConv, global_mean_pool

class MyGraphCLIPModel(nn.Module):
    """
    Sistemin ana mimarisi: GNN ve CLIP tabanlı uzamsal akıl yürütme modeli.
    """
    def __init__(self, node_feature_dim=512, hidden_dim=512, output_dim=512):
        super(MyGraphCLIPModel, self).__init__()
        
        # 1. GNN Modülü: Nesneler arası uzamsal ilişkileri (Message Passing) öğrenir.
        # Graph Attention Network (GAT) kullanıyoruz.
        self.gnn_layer1 = GATConv(in_channels=node_feature_dim, out_channels=hidden_dim, heads=4, concat=False)
        self.gnn_layer2 = GATConv(in_channels=hidden_dim, out_channels=output_dim, heads=1, concat=False)
        
        # Aktivasyon ve Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 2. Fusion (Birleştirme) Modülü (Opsiyonel/İleri Seviye)
        # GNN'den çıkan uzamsal veriyi, genel görüntü özellikleriyle birleştirmek için bir MLP.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim), # *2: Biri GNN'den, biri CLIP'ten geliyorsa
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node_features, edge_index, batch_index=None):
        """
        İleri besleme (Forward Pass)
        
        Args:
            node_features: (N, 512) boyutunda, her nesnenin başlangıç özellikleri .
            edge_index: (2, M) boyutunda, düğümler arası bağlantılar .
            batch_index: Birden fazla grafi aynı anda işlemek için.
        """
        
        # Aşama 1: Message Passing (Nesneler birbirlerine "ben buradayım" der)
        x = self.gnn_layer1(node_features, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.gnn_layer2(x, edge_index)
        
        # Şu an 'x', her bir nesnenin (düğümün) 'uzamsal farkındalığı olan' yeni vektörlerini içeriyor.
        
        # Aşama 2: Grafı Tek Bir Vektöre İndirgeme (Readout / Global Pooling)
        # Tüm nesnelerin bilgisini tek bir "Sahne" vektörü olacak şekilde birleştiriyoruz.
        if batch_index is None:
            # Tek bir görüntü işliyorsak, basitçe tüm düğümlerin ortalamasını alabiliriz.
            # (N, 512) -> (1, 512)
            graph_embedding = torch.mean(x, dim=0, keepdim=True) 
        else:
            # Batch ile çalışıyorsak PyG'nin havuzlama fonksiyonunu kullanırız.
            graph_embedding = global_mean_pool(x, batch_index)
            
        # Aşama 3: (Opsiyonel) Fusion
        # Burada graph_embedding'i, doğrudan OpenCLIP'ten gelen genel görüntü 
        # vektörü ile birleştirebilir (fusion_mlp) ve sonucu dönebilirsin.
        
        # Şimdilik sadece uzamsal farkındalığı olan GNN embedding'ini dönüyoruz.
        return graph_embedding