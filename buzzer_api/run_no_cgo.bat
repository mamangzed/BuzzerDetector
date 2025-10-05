@echo off
echo Building and running Buzzer API without CGO...
set CGO_ENABLED=0
go run main.go handlers.go missing_handlers.go