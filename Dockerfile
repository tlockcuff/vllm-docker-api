# Use Node.js 18 as base image
FROM node:18-alpine

# Install Python and pip for huggingface-cli
RUN apk add --no-cache python3 py3-pip

# Install huggingface-cli with transfer acceleration
RUN pip3 install --break-system-packages huggingface_hub[hf-transfer]

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Create models directory
RUN mkdir -p models

RUN npm run build

# Expose port
EXPOSE 3000

# Run the application
CMD ["npm", "start"]
