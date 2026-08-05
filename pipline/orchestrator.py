
from core.interfaces import IDataParser, IVisionExtractor

class GraphCLIPPipeline:
    def __init__(self, parser: IDataParser, extractor: IVisionExtractor):
        self.parser = parser
        self.extractor = extractor

    def prepare_data(self, image_id: str):
        
        graph = self.parser.parse_image(image_id)
        rich_graph = self.extractor.extract_features(graph)
        return rich_graph