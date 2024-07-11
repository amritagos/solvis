#!/bin/sh
find ./solvis -regex '.*py' -exec black {} \;
find ./tests -regex '.*py' -exec black {} \;
find ./examples -regex '.*py' -exec black {} \;