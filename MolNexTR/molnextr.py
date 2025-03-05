import logging
import torch
import pystow
import os

from MolNexTR.model import molnextr

logging.getLogger("absl").setLevel("ERROR")
device = torch.device("cpu")

# Set path
default_path = pystow.join("molnextr")

model_url = (
    "https://huggingface.co/datasets/CYF200127/MolNexTR/resolve/main/molnextr_best.pth"
)


def download_trained_weights(model_url: str, model_path: str, verbose: int = 1) -> str:
    """
    Download the trained model from the specified URL and save it to the specified path.

    If the model already exists, it will not be downloaded again.

    Args:
        model_url (str): The URL to download the model from.
        model_path (str): The path where the model should be saved.
        verbose (int, optional): Verbosity level (0 = silent, 1 = detailed logs). Defaults to 1.

    Returns:
        str: Path to the downloaded model file.
    """
    if verbose > 0:
        print("Downloading trained model to " + str(model_path))
    model_path = pystow.ensure("molnextr", url=model_url)
    if verbose > 0:
        print(model_path)
        print("... done downloading trained model!")

    return model_path


def ensure_model(default_path: str, model_url: str = model_url) -> str:
    """
    Ensure the model is available locally by downloading it if necessary.

    Args:
        default_path (str): The base path where the model will be stored.
        model_url (str, optional): The URL to download the model from. Defaults to the pre-defined `model_url`.

    Returns:
        str: Path to the downloaded model file.
    """
    model_download_path = os.path.join(default_path.as_posix(), "MolNexTR_model")
    model_path = download_trained_weights(model_url, default_path)
    return model_path


model_path = ensure_model(default_path=default_path, model_url=model_url)
MolNexTR_model = molnextr(model_path, device)


def get_predictions(
    imagepath: str,
    atoms_bonds: bool = False,
    smiles: bool = True,
    predicted_molfile: bool = False,
    model=MolNexTR_model,
):
    """
    Generate predictions from the MolNexTR model for a given chemical structure image.
    This function processes the input image and returns predictions in various formats,
    including SMILES strings, molecular files, or atom-bond mappings.
    Args:
        imagepath (str): Path to the input image containing the chemical structure.
        atoms_bonds (bool, optional): Whether to return atom-bond mappings in the predictions. Defaults to False.
        smiles (bool, optional): Whether to return the predicted SMILES string. Defaults to True.
        predicted_molfile (bool, optional): Whether to return the predicted molecular file. Defaults to False.
        model: The MolNexTR model instance to use for predictions. Defaults to `MolNexTR_model`.
    Returns:
        dict: The prediction results. Returns a dictionary containing the requested output formats
              based on the boolean flags provided.
    """
    # Get predictions with atoms_bonds if requested
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
    
    # If no specific outputs were requested, return all available predictions
    if not (smiles or atoms_bonds or predicted_molfile):
        return predictions
    
    # Return the result dictionary with all requested outputs
    return result
