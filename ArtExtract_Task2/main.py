import argparse
from dataset import DatasetLoader 
from image_downloader import ImageDownloader
from model import Model  
from image_retrieval import ImageRetrieval
from evaluation import Evaluator 

def main(args):
    dataset = DatasetLoader(args.dataset_path)
    dataset.load_data()
    
    downloader = ImageDownloader(args.image_dir)
    downloader.get_base_dir()
    
    model = Model()
    model.load_model(args.model_path)
    
    retriever = ImageRetrieval(model, dataset)
    results = retriever.retrieve_images(args.query_image)
    
    evaluator = Evaluator()
    score = evaluator.evaluate(results, args.ground_truth)
    print(f"Evaluation Score: {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory to store downloaded images")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--query_image", type=str, required=True, help="Path to the query image for retrieval")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth data for evaluation")
    
    args = parser.parse_args()
    main(args)