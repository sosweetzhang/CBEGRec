"""Download BERT model (bert-base-chinese) for local use."""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_bert():
    """Download BERT model."""
    print("=" * 60)
    print("BERT Model Download Script")
    print("=" * 60)
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print(f"\nUsing HF Mirror: {os.environ['HF_ENDPOINT']}")
    
    model_name = "bert-base-chinese"
    print(f"\nDownloading BERT model: {model_name}")
    print("This may take several minutes depending on your network speed...")
    
    try:
        from transformers import BertModel, BertTokenizer
        
        print("\n[1/2] Downloading BERT tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer downloaded successfully")
        
        print("\n[2/2] Downloading BERT model...")
        model = BertModel.from_pretrained(model_name)
        print("✅ Model downloaded successfully")
        
        print("\nTesting model...")
        test_text = "test text"
        inputs = tokenizer(test_text, return_tensors='pt')
        outputs = model(**inputs)
        print(f"✅ Model test passed. Output shape: {outputs.last_hidden_state.shape}")
        
        print(f"\n✅ BERT model '{model_name}' downloaded and verified successfully!")
        print(f"\nModel will be cached at: ~/.cache/huggingface/hub/")
        print(f"You can now use 'bert_path: \"{model_name}\"' in config.yaml")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Error downloading BERT:")
        print(f"   {type(e).__name__}: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check your network connection")
        print(f"2. Verify HF_ENDPOINT is set correctly: {os.environ.get('HF_ENDPOINT')}")
        print(f"3. Try manually downloading from: https://hf-mirror.com/models/{model_name}")
        print(f"4. Check firewall/antivirus settings")
        return False

if __name__ == "__main__":
    success = download_bert()
    sys.exit(0 if success else 1)
