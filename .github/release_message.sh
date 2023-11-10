#!/usr/bin/env bash
previous_tag=$(git tag --sort=-creatordate | sed -n 2p)
git shortlog | sed 's/^./    &/'  # change back to 'git shortlog "${previous_tag}.."' when there are more tags
