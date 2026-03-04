#!/bin/bash
set -e

# Check if a bump type is provided
if [ -z "$1" ]; then
    echo "Usage: ./bump_version.sh <patch|minor|major>"
    exit 1
fi

BUMP_TYPE=$1

# Validate bump type
if [[ ! "$BUMP_TYPE" =~ ^(patch|minor|major)$ ]]; then
    echo "Error: Bump type must be 'patch', 'minor', or 'major'"
    exit 1
fi

# Get current version from pyproject.toml
CURRENT_VERSION=$(sed -nE 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' pyproject.toml)

if [ -z "$CURRENT_VERSION" ]; then
    echo "Error: Could not find version in pyproject.toml"
    exit 1
fi

# Display current version
echo "Current version: $CURRENT_VERSION"

# Parse version components
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

# Calculate new version
case $BUMP_TYPE in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
esac

# Print new version
NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "New version: $NEW_VERSION"
echo ""

# Check if on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Error: You must be on the main branch"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: You have uncommitted changes"
    git status --short
    exit 1
fi

# Run tests
echo "Running tests..."
pytest -q --maxfail=1
echo ""

# Testing documentation
echo "Testing documentation build..."
if [ ! -d "docs/src" ]; then
    echo "Error: docs/src directory not found"
    exit 1
fi

pip install -q -r docs/src/requirements.txt 2>/dev/null || true

cd docs/src
rm -rf _build
make html > /tmp/sphinx_build.log 2>&1
if grep -qi "error" /tmp/sphinx_build.log; then
    echo "Error: Documentation build failed"
    cat /tmp/sphinx_build.log
    exit 1
fi
rm -r _build
cd ../..

echo "Documentation built successfully"
echo ""

# Update version numbers in pyproject.toml and flowtube/__init__.py
echo "Updating version numbers..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
    sed -i '' "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" flowtube/__init__.py
else
    sed -i "s/version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
    sed -i "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" flowtube/__init__.py
fi

# Print success message and next steps
echo "Version updated to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  git add pyproject.toml flowtube/__init__.py"
echo "  git commit -m 'Bump version to $NEW_VERSION'"
echo "  git push origin main"
echo "  git tag v$NEW_VERSION"
echo "  git push origin v$NEW_VERSION"