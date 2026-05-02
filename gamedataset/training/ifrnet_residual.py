import sys
from gamedataset.training.common import main

if __name__ == "__main__":
    if "--model-name" not in sys.argv:
        sys.argv.extend(["--model-name", "IFRNet_Residual"])
    main()
