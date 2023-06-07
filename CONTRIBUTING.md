# Contributing to Autodistill 🛠️

Thank you for your interest in contributing to Autodistill!

We welcome any contributions to help us improve the quality of `autodistill` and expand the range of supported models.

## Contribution Guidelines

We welcome contributions to:

1. Add a new base model (see more guidance below).
2. Add a new target model (see more guidance below).
3. Report bugs and issues in the project.
4. Submit a request for a new task or feature.
5. Improve our test coverage.

### Contributing Features

Autodistill is designed with modularity in mind. We want `autodistill` to extend across different models and problem types, providing a consistent interface for distilling models.

We welcome contributions that add new models to the project. Before you begin, please make sure that another contributor has not already begun work on the model you want to add. You can check the [project README](https://github.com/autodistill/autodistill/blob/main/README.md) for our roadmap on adding more models.

To add a new model, create a new repo that requires `autodistill` and implement the `BaseModel` or `TargetModel` class for your task. You can use the existing models as a guide for how to structure your code.

Finally, you will need to add documentation for your model and link to it from the `autodistill` README. You can add a new page to the `docs/models` directory that describes your model and how to use it. You can use the existing model documentation as a guide for how to structure your documentation.

## How to Contribute Changes

First, fork this repository to your own GitHub account. Create a new branch that describes your changes (i.e. `line-counter-docs`). Push your changes to the branch on your fork and then submit a pull request to this repository.

When creating new functions, please ensure you have the following:

1. Docstrings for the function and all parameters.
2. Examples in the documentation for the function.
3. Created an entry in our docs to autogenerate the documentation for the function.

All pull requests will be reviewed by the maintainers of the project. We will provide feedback and ask for changes if necessary.

PRs must pass all tests and linting requirements before they can be merged.

## 🧹 Code quality 

We provide two handy commands inside the `Makefile`, namely:

- `make style` to format the code
- `make check_code_quality` to check code quality (PEP8 basically)

So far, **there is no types checking with mypy**. See [issue](https://github.com/roboflow/template-python/issues/4). 

## 🧪 Tests 

[`pytests`](https://docs.pytest.org/en/7.1.x/) is used to run our tests.