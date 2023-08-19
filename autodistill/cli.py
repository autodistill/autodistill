from autodistill.detection import CaptionOntology
from autodistill.registry import import_requisite_module
import os
import click

@click.command()
@click.argument('dir')
@click.option('--base', default='base', required=True)
@click.option('--target', default='target', required=True)
@click.option('--ontology', default={}, required=True)
@click.option("--epochs", default=200, required=True)
@click.option("--output", default="./dataset", required=True)
def main(dir, base, target, ontology, epochs, output):
    print("Loading base model...")
    model = import_requisite_module(base)
    ontology = CaptionOntology(ontology)
    base_model = model(ontology=ontology)

    print("Labeling data...")

    base_model.label(
        input_folder=dir,
        output_folder=output
    )

    print("Loading target model...")
    target_model = import_requisite_module(target)

    print("Training target model...")
    target_model.train(os.path.join(output, "data.yaml"), epochs=epochs)

    print("✨ Your model has been trained! ✨")
    
    
if __name__ == '__main__':
    main()
