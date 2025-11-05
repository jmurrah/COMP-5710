## COMP-5710 Workshop 9

Each of the 5 files are test inputs designed to trigger errors in the `FileOpener` binary. There is a single `crashes.png` screenshot showing all the errors.

### Directions

1. Download the workshop 9 Docker image.
   ```bash
   docker pull akondrahman/fuzz4life
   ```
2. Start the container and an interactive terminal for the image.
   ```bash
   docker run --rm -it --name workshop9 akondrahman/fuzz4life:latest bash
   ```
3. Copy the input files (all files are in the local `inputs/` folder) into the running Docker container.
   ```bash
   docker cp inputs/. workshop9:/WORKSHOP/blackbox/
   ```
4. Navigate to the FileOpener binary file.
   ```bash
   cd /WORKSHOP/blackbox
   ```
5. For each file, run:
   ```bash
   ./dist/FileOpener <filename>
   ```
   NOTE: Replace `<filename>` with the actual filename (e.g., `input1-utf8codec.txt`, `input2-dir`, etc.).
