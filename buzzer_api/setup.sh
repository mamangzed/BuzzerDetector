#!/bin/bash

# Buzzer Detection API Setup Script

echo "Setting up Buzzer Detection API..."

# Initialize Go module if not exists
if [ ! -f "go.mod" ]; then
    echo "Initializing Go module..."
    go mod init buzzer-api
fi

# Install dependencies
echo "Installing Go dependencies..."
go get github.com/gin-gonic/gin@v1.9.1
go get github.com/mattn/go-sqlite3@v1.14.17
go get github.com/swaggo/gin-swagger@v1.6.0
go get github.com/swaggo/files@v1.0.1
go get github.com/swaggo/swag/cmd/swag@v1.16.2

# Generate swagger documentation
echo "Generating Swagger documentation..."
swag init

# Build the application
echo "Building the application..."
go build -o buzzer-api main.go

echo "Setup complete!"
echo ""
echo "To run the API:"
echo "  ./buzzer-api"
echo ""
echo "API will be available at:"
echo "  - Main API: http://localhost:8080/api/v1/"
echo "  - Swagger docs: http://localhost:8080/swagger/index.html"
echo "  - Health check: http://localhost:8080/health"