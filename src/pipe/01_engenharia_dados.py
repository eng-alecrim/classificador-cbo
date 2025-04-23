from core.data.bronze import main as bronze_layer_processing
from core.data.silver import main as silver_layer_processing


def main() -> None:
    bronze_layer_processing()
    silver_layer_processing()
    return None


if __name__ == "__main__":
    main()
