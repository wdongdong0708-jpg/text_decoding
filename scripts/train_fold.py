"""Single-fold entry point.

Stage 6A intentionally delegates to the guarded smoke runner. Formal long
training must use a separately frozen configuration in the next stage.
"""

from smoke_fold_training import main


if __name__ == "__main__":
    main()
