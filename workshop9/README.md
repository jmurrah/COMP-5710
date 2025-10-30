## COMP-5710 Workshop 9

Each of the 5 `.txt` files are test inputs designed to trigger errors in the `FileOpener` binary.  
For each input file, there is a corresponding screenshot that shows the resulting crash.  
For example:  
- `input1.txt` → `input1crash.png`
- `input2.txt` → `input2crash.png`

### Directions
1. Download the workshop 9 Docker image.
   ```bash
   docker pull akondrahman/fuzz4life
   ```
2. Start the container and an interactive terminal for the image.
   ```bash
   docker run --rm -it --name workshop9 akondrahman/fuzz4life:latest bash
   ```
3. Copy the input .txt files into the running Docker container.
   ```bash
   docker cp input1.txt input2.txt input3.txt input4.txt input5.txt workshop9:/WORKSHOP/blackbox/
   ```
4. Navigate to the FileOpener binary file.
   ```bash
   cd /WORKSHOP/blackbox
   ```
5. For each `.txt` file, run:
   ```bash
   ./dist/FileOpener myfile.txt
   ```
   NOTE: Replace `myfile.txt` with the actual filename (e.g., `input1.txt`, `input2.txt`, etc.).