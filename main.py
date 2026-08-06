import torch

# Test için ilk üretilen .pt dosyasını yüklüyoruz
test_graph = torch.load("processed_data/2.pt")

print("--- .pt Dosyası Sağlık Raporu ---")
print(f"Resim ID: {test_graph.image_id}")
print(f"Toplam Nesne (Node) Sayısı: {len(test_graph.nodes)}")

if len(test_graph.nodes) > 0:
    ilk_node = test_graph.nodes[0]
    print(f"İlk Düğüm Etiketi/İsmi: {getattr(ilk_node, 'names', 'Bilinmiyor')}")
    print(f"Feature Tensor Boyutu (Shape): {ilk_node.feature_tensor.shape}")
    # Beklenen çıktı torch.Size([512]) olmalıdır.
    
edge_index = test_graph.get_edge_index()
print(f"Edge Index Boyutu (Shape): {edge_index.shape}")
print("-----------------------------------")