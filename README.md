# SVD Image Compression

Compress images using low-rank approximations. A higher *k* will result in better approximations but requires more data to encode the image.

rank = 5, rank = 25, rank = 100

## Usage

```
$ cargo run <path_to_image>
```

Note that large images will take longer to compress.
