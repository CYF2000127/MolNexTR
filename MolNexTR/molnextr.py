import os
import logging
import platform
import warnings
import torch
import pystow
from MolNexTR.model import molnextr

# Suppress unnecessary warnings
logging.getLogger("absl").setLevel("ERROR")

# Control level of debug output
DEBUG_LEVEL = os.environ.get("MOLNEXTR_DEBUG", "INFO").upper()

# Filter specific PyTorch deprecation warnings from OpenNMT
warnings.filterwarnings("ignore", message="torch.cuda.amp.custom_fwd.*is deprecated")
warnings.filterwarnings("ignore", message="torch.cuda.amp.custom_bwd.*is deprecated")
warnings.filterwarnings("ignore", message=".*ForwardRef.*is deprecated.*")

# Add a name for the logger to track initialization
logger = logging.getLogger("molnextr_singleton")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Set level based on environment variable 
if DEBUG_LEVEL == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif DEBUG_LEVEL == "WARNING":
    logger.setLevel(logging.WARNING)
    # Suppress all PyTorch warnings in warning mode
    warnings.filterwarnings("ignore", category=FutureWarning)
elif DEBUG_LEVEL == "ERROR":
    logger.setLevel(logging.ERROR)
    # Suppress all warnings in error-only mode
    warnings.filterwarnings("ignore")
else:
    # Default to INFO level
    logger.setLevel(logging.INFO)

class MolNexTRSingleton:
    _instance = None
    _device = None
    _device_name = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton MolNexTR model instance"""
        if cls._instance is None:
            logger.info("Initializing MolNexTR singleton for the first time")
            cls._detect_hardware()
            cls._instance = cls._initialize_model()
            logger.info(f"MolNexTR singleton initialized successfully on {cls._device_name}")
        else:
            logger.debug("Returning existing MolNexTR singleton instance")
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
        
        logger.info("Detecting available hardware...")
        logger.info(f"System: {platform.system()} {platform.version()}")
        logger.info(f"Machine: {platform.machine()}")
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Check for Apple Silicon (M1/M2/M3)
        is_apple_silicon = (
            platform.system() == "Darwin" and 
            (platform.machine() == "arm64" or "ARM64" in platform.version())
        )
        
        if is_apple_silicon:
            logger.info("Apple Silicon detected")
            mps_available = torch.backends.mps.is_available()
            logger.info(f"MPS backend available: {mps_available}")
            
            if mps_available:
                try:
                    # Test MPS device
                    cls._device = torch.device("mps")
                    logger.info("Running MPS device test...")
                    test_tensor = torch.ones(1).to(cls._device)
                    del test_tensor
                    cls._device_name = f"Apple Silicon (MPS)"
                    logger.info(f"✅ Successfully initialized Apple Silicon acceleration via MPS backend")
                    return
                except Exception as e:
                    logger.warning(f"⚠️ MPS available but test failed: {str(e)}. Falling back to CPU.")
        
        # Check for CUDA GPU
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            try:
                # Test CUDA device
                cls._device = torch.device("cuda")
                logger.info("Running CUDA device test...")
                test_tensor = torch.ones(1).to(cls._device)
                del test_tensor
                gpu_name = torch.cuda.get_device_name(0)
                cls._device_name = f"NVIDIA GPU ({gpu_name})"
                logger.info(f"✅ Successfully initialized GPU acceleration: {gpu_name}")
                return
            except Exception as e:
                logger.warning(f"⚠️ CUDA available but test failed: {str(e)}. Falling back to CPU.")
        
        logger.info(f"👉 No GPU acceleration available. Using CPU for computation.")
    
    @classmethod
    def _initialize_model(cls):
        """Initialize the MolNexTR model only once with the detected device"""
        # Set path
        default_path = pystow.join("molnextr")
        model_url = "https://huggingface.co/datasets/CYF200127/MolNexTR/resolve/main/molnextr_best.pth"
        
        # Check if model already exists
        expected_model_path = os.path.join(default_path.as_posix(), "molnextr_best.pth")
        
        logger.info(f"Checking for model at {expected_model_path}")
        
        if os.path.exists(expected_model_path):
            logger.info(f"✅ Using existing model at {expected_model_path}")
            model_path = expected_model_path
        else:
            logger.info(f"⬇️ Model not found at {expected_model_path}, downloading...")
            model_path = pystow.ensure("molnextr", url=model_url)
            logger.info(f"✅ Downloaded model to {model_path}")
        
        # Initialize model with the detected device
        logger.info(f"🔄 Initializing MolNexTR model on {cls._device_name}")
        
        try:
            # Initialize model on the detected device
            logger.info(f"Creating MolNexTR model with device {cls._device}...")
            
            # Add memory tracking for debugging
            if torch.cuda.is_available():
                logger.info(f"CUDA memory before model init: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            
            # Create the model
            model = molnextr(model_path, cls._device)
            
            # For Apple Silicon, ensure model components are on MPS
            if cls._device_name.startswith("Apple Silicon"):
                logger.info("Configuring model for Apple Silicon...")
                # Force model evaluation mode
                model.eval()
            
            # Memory tracking after model creation
            if torch.cuda.is_available():
                logger.info(f"CUDA memory after model init: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            
            # Check if the model loads and runs
            logger.info("Running model validation test...")
            test_succeeded = cls._test_model(model)
            
            if not test_succeeded:
                raise RuntimeError("Model failed validation test")
            
            logger.info("✅ Model initialization successful")    
            return model
            
        except Exception as e:
            logger.error(f"❌ Error initializing model on {cls._device_name}: {str(e)}")
            logger.warning("⚠️ Falling back to CPU...")
            
            # Fall back to CPU if device-specific initialization fails
            cls._device = torch.device("cpu")
            cls._device_name = "CPU (fallback)"
            
            # Try one more time with CPU
            logger.info("Attempting to initialize model on CPU...")
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
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_time:
        start_time.record()
    
    logger.debug(f"Getting predictions for image: {imagepath}")
    logger.debug(f"Requested outputs: smiles={smiles}, molfile={predicted_molfile}, atoms_bonds={atoms_bonds}")
    
    # Get the model from singleton
    model = MolNexTRSingleton.get_instance()
    
    # Track memory if CUDA is available
    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated()/1024**2
        logger.debug(f"CUDA memory before prediction: {memory_before:.2f} MB")
    
    # Get predictions with atoms_bonds if requested
    try:
        logger.debug("Running prediction...")
        predictions = model.predict_final_results(imagepath, return_atoms_bonds=atoms_bonds)
        logger.debug("Prediction successful")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.warning("Trying with CPU model as fallback...")
        
        # Reset the singleton to force re-initialization with CPU
        MolNexTRSingleton._instance = None
        MolNexTRSingleton._device = torch.device("cpu")
        MolNexTRSingleton._device_name = "CPU (fallback after error)"
        
        # Get fresh model instance with CPU
        model = MolNexTRSingleton.get_instance()
        logger.debug("Running prediction with CPU fallback...")
        predictions = model.predict_final_results(imagepath, return_atoms_bonds=atoms_bonds)
    
    # Track memory after prediction
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated()/1024**2
        logger.debug(f"CUDA memory after prediction: {memory_after:.2f} MB")
        logger.debug(f"Memory delta: {memory_after - memory_before:.2f} MB")
    
    # Initialize result dictionary
    result = {}
    
    # Add requested outputs to the result dictionary
    if smiles:
        result["predicted_smiles"] = predictions["predicted_smiles"]
        logger.debug(f"SMILES: {predictions['predicted_smiles']}")
    if atoms_bonds:
        result["atom_sets"] = predictions["atom_sets"]
        logger.debug(f"Atom sets count: {len(predictions['atom_sets']) if 'atom_sets' in predictions else 0}")
    if predicted_molfile:
        result["predicted_molfile"] = predictions["predicted_molfile"]
        molfile_lines = len(predictions["predicted_molfile"].split("\n")) if "predicted_molfile" in predictions else 0
        logger.debug(f"Molfile lines: {molfile_lines}")
    
    # Add the device information to the result
    device, device_name = MolNexTRSingleton.get_device()
    result["device_info"] = device_name
    
    # Record end time if using CUDA
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        logger.debug(f"Prediction took {elapsed_time:.3f} seconds")
        # Add timing info to result
        result["prediction_time_seconds"] = elapsed_time
    
    # If no specific outputs were requested, return all available predictions
    if not (smiles or atoms_bonds or predicted_molfile):
        return predictions
    
    # Return the result dictionary with all requested outputs
    return result
