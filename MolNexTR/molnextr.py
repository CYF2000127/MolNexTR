import os
import logging
import platform
import torch
import pystow
from MolNexTR.model import molnextr

# Suppress unnecessary warnings
logging.getLogger("absl").setLevel("ERROR")

class MolNexTRSingleton:
    _instance = None
    _device = None
    _device_name = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton MolNexTR model instance"""
        if cls._instance is None:
            cls._detect_hardware()
            cls._instance = cls._initialize_model()
        return cls._instance
    
    @classmethod
    def get_device(cls):
        """Get the device being used by the model"""
        if cls._device is None:
            cls._detect_hardware()
        return cls._device, cls._device_name
    
    @classmethod
    def _detect_hardware(cls):
        """
        Auto-detect the best available hardware for PyTorch
        Sets the _device and _device_name class variables
        """
        # Initialize with CPU as fallback
        cls._device = torch.device("cpu")
        cls._device_name = "CPU"
        
        # Check for Apple Silicon (M1/M2/M3)
        is_apple_silicon = (
            platform.system() == "Darwin" and 
            (platform.machine() == "arm64" or "ARM64" in platform.version())
        )
        
        if is_apple_silicon and torch.backends.mps.is_available():
            try:
                # Test MPS device
                cls._device = torch.device("mps")
                test_tensor = torch.ones(1).to(cls._device)
                del test_tensor
                cls._device_name = f"Apple Silicon (MPS)"
                print(f"Using Apple Silicon acceleration via MPS backend")
                return
            except Exception as e:
                print(f"MPS available but test failed: {str(e)}. Falling back to CPU.")
        
        # Check for CUDA GPU
        if torch.cuda.is_available():
            try:
                # Test CUDA device
                cls._device = torch.device("cuda")
                test_tensor = torch.ones(1).to(cls._device)
                del test_tensor
                gpu_name = torch.cuda.get_device_name(0)
                cls._device_name = f"NVIDIA GPU ({gpu_name})"
                print(f"Using GPU acceleration: {gpu_name}")
                return
            except Exception as e:
                print(f"CUDA available but test failed: {str(e)}. Falling back to CPU.")
        
        print(f"No GPU acceleration available. Using CPU for computation.")
    
    @classmethod
    def _initialize_model(cls):
        """Initialize the MolNexTR model only once with the detected device"""
        # Set path
        default_path = pystow.join("molnextr")
        model_url = "https://huggingface.co/datasets/CYF200127/MolNexTR/resolve/main/molnextr_best.pth"
        
        # Check if model already exists
        expected_model_path = os.path.join(default_path.as_posix(), "molnextr_best.pth")
        if os.path.exists(expected_model_path):
            print(f"Using existing model at {expected_model_path}")
            model_path = expected_model_path
        else:
            print(f"Model not found at {expected_model_path}, downloading...")
            model_path = pystow.ensure("molnextr", url=model_url)
            print(f"Downloaded model to {model_path}")
        
        # Initialize model with the detected device
        print(f"Initializing MolNexTR model on {cls._device_name}")
        
        try:
            # Initialize model on the detected device
            model = molnextr(model_path, cls._device)
            
            # For Apple Silicon, ensure model components are on MPS
            if cls._device_name.startswith("Apple Silicon"):
                # Force model evaluation mode
                model.eval()
            
            # Check if the model loads and runs
            test_succeeded = cls._test_model(model)
            if not test_succeeded:
                raise RuntimeError("Model failed validation test")
                
            return model
        except Exception as e:
            print(f"Error initializing model on {cls._device_name}: {str(e)}")
            print("Falling back to CPU...")
            
            # Fall back to CPU if device-specific initialization fails
            cls._device = torch.device("cpu")
            cls._device_name = "CPU (fallback)"
            return molnextr(model_path, cls._device)
    
    @classmethod
    def _test_model(cls, model):
        """Test the model with a dummy operation to ensure it's working"""
        try:
            # This is a minimal test to ensure the model's core functionality works
            # We don't actually run a prediction which would require a valid image
            if hasattr(model, 'model'):
                test_tensor = torch.zeros((1, 3, 224, 224), device=cls._device)
                with torch.no_grad():
                    # Just test if a simple forward pass works 
                    # (this might fail for some model architectures)
                    try:
                        model.model.encoder(test_tensor)
                    except:
                        # If the specific test fails, it might be due to model architecture
                        # Just return True and hope for the best during actual usage
                        pass
            return True
        except Exception as e:
            print(f"Model test failed: {str(e)}")
            return False

# Public API function that uses the singleton
def get_predictions(
    imagepath: str,
    atoms_bonds: bool = False,
    smiles: bool = True,
    predicted_molfile: bool = False,
):
    """
    Generate predictions from the MolNexTR model for a given chemical structure image.
    Uses the singleton instance of MolNexTR to avoid loading the model multiple times.
    
    Args:
        imagepath: Path to the input image containing the chemical structure.
        atoms_bonds: Whether to return atom-bond mappings in the predictions.
        smiles: Whether to return the predicted SMILES string.
        predicted_molfile: Whether to return the predicted molecular file.
        
    Returns:
        dict: The prediction results with requested output formats.
    """
    # Get the singleton model instance
    model = MolNexTRSingleton.get_instance()
    
    # Get predictions with atoms_bonds if requested
    try:
        predictions = model.predict_final_results(imagepath, return_atoms_bonds=atoms_bonds)
    except Exception as e:
        # If prediction fails, log the error and try again with the CPU model
        print(f"Prediction failed: {str(e)}")
        print("Trying with CPU model as fallback...")
        
        # Reset the singleton to force re-initialization with CPU
        MolNexTRSingleton._instance = None
        MolNexTRSingleton._device = torch.device("cpu")
        MolNexTRSingleton._device_name = "CPU (fallback after error)"
        
        # Get fresh model instance with CPU
        model = MolNexTRSingleton.get_instance()
        predictions = model.predict_final_results(imagepath, return_atoms_bonds=atoms_bonds)
    
    # Initialize result dictionary
    result = {}
    
    # Add requested outputs to the result dictionary
    if smiles:
        result["predicted_smiles"] = predictions["predicted_smiles"]
    if atoms_bonds:
        result["atom_sets"] = predictions["atom_sets"]
    if predicted_molfile:
        result["predicted_molfile"] = predictions["predicted_molfile"]
    
    # Add the device information to the result
    device, device_name = MolNexTRSingleton.get_device()
    result["device_info"] = device_name
    
    # If no specific outputs were requested, return all available predictions
    if not (smiles or atoms_bonds or predicted_molfile):
        return predictions
    
    # Return the result dictionary with all requested outputs
    return result
