# Buzzer Detection API

This is a REST API built with Go and Gin framework for searching and analyzing buzzer accounts detected by the Python buzzer detection system.

## Features

- **Account Search**: Search buzzer accounts by username, platform, risk level, and probability threshold
- **Post Search**: Search posts by content, username, and platform
- **Coordination Analysis**: View coordination groups and their members
- **Statistics**: Get comprehensive statistics about detection results
- **Real-time Data**: Direct connection to SQLite database updated by Python detection system
- **Swagger Documentation**: Interactive API documentation
- **CORS Support**: Cross-origin resource sharing for web applications

## API Endpoints

### Accounts
- `GET /api/v1/accounts` - Search buzzer accounts with filters
- `GET /api/v1/accounts/{username}` - Get specific account details
- `GET /api/v1/accounts/platform/{platform}` - Get accounts by platform
- `GET /api/v1/accounts/risk/{level}` - Get accounts by risk level

### Posts
- `GET /api/v1/posts` - Search posts with filters
- `GET /api/v1/posts/user/{username}` - Get posts by user
- `GET /api/v1/posts/platform/{platform}` - Get posts by platform

### Coordination
- `GET /api/v1/coordination/groups` - Get coordination groups
- `GET /api/v1/coordination/group/{groupId}` - Get group details

### Statistics
- `GET /api/v1/stats` - Overall statistics
- `GET /api/v1/stats/platforms` - Platform statistics
- `GET /api/v1/stats/risks` - Risk level statistics

### Search
- `GET /api/v1/search` - Global search across all entities

### Health
- `GET /health` - Health check endpoint

## Quick Start

### Prerequisites
- Go 1.21 or higher
- SQLite3
- Python buzzer detection system (to generate data)

### Setup

1. **Windows:**
   ```cmd
   cd buzzer_api
   setup.bat
   ```

2. **Linux/Mac:**
   ```bash
   cd buzzer_api
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Manual setup:**
   ```bash
   go mod tidy
   go get github.com/gin-gonic/gin
   go get github.com/mattn/go-sqlite3
   go get github.com/swaggo/gin-swagger
   go get github.com/swaggo/files
   go build -o buzzer-api main.go
   ```

### Run

```bash
./buzzer-api        # Linux/Mac
buzzer-api.exe      # Windows
```

The API will start on `http://localhost:8080`

## Usage Examples

### Search High-Risk Accounts
```bash
curl "http://localhost:8080/api/v1/accounts?risk_level=HIGH&page=1&page_size=10"
```

### Search Accounts by Platform
```bash
curl "http://localhost:8080/api/v1/accounts/platform/x"
```

### Search Posts with Content
```bash
curl "http://localhost:8080/api/v1/posts?content=buzzer&page=1"
```

### Get Statistics
```bash
curl "http://localhost:8080/api/v1/stats"
```

### Global Search
```bash
curl "http://localhost:8080/api/v1/search?q=username&type=all"
```

## Response Format

All endpoints return JSON responses with consistent structure:

### Search Responses
```json
{
  "data": [...],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "total_pages": 5
}
```

### Error Responses
```json
{
  "error": "Error message"
}
```

## Data Model

### BuzzerAccount
- **id**: Unique identifier
- **username**: Account username
- **social_media_platform**: Platform (x, twitter, etc.)
- **account_url**: Direct link to account
- **posts_count**: Number of posts analyzed
- **followers_count**: Follower count
- **freq_per_day**: Posting frequency per day
- **burstiness**: Posting pattern burstiness score
- **buzzer_prob_enhanced**: Enhanced buzzer probability (0-1)
- **risk_category**: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
- **coord_cluster**: Coordination group ID (-1 if not coordinated)

### Post
- **post_id**: Original post ID from platform
- **username**: Post author
- **content**: Post text content
- **post_url**: Direct link to post
- **retweet_count**, **like_count**: Engagement metrics
- **is_retweet**: Boolean indicating if post is a retweet

### CoordinationGroup
- **group_id**: Group identifier
- **username**: Member username
- **similarity_score**: Coordination similarity score

## Configuration

### Database Path
By default, the API looks for `buzzer_detection.db` in the parent directory. To change this, modify the database path in `main.go`:

```go
db, err = sql.Open("sqlite3", "path/to/your/database.db")
```

### Port Configuration
To change the port from 8080, modify the `Run` call in `main.go`:

```go
r.Run(":9000")  // Use port 9000
```

## Integration with Python System

The API automatically reads from the SQLite database created by the Python buzzer detection system. Make sure to:

1. Run the Python detection first to generate data:
   ```bash
   python main.py --mode playwright --query "test" --max_posts 100 --out results
   ```

2. The database file `buzzer_detection.db` will be created automatically

3. Start the API to serve the results

## Development

### Add New Endpoints
1. Define handler function
2. Add route in `main()` function
3. Add Swagger documentation comments
4. Rebuild: `go build main.go`

### Database Schema
The API expects these tables:
- `buzzer_accounts`: Main account data
- `posts`: Social media posts
- `coordination_groups`: Group membership data

## Swagger Documentation

Interactive API documentation is available at:
`http://localhost:8080/swagger/index.html`

This provides:
- Complete endpoint documentation
- Request/response examples
- Interactive testing interface
- Model definitions

## Performance Notes

- Uses SQLite with proper indexing for fast queries
- Supports pagination for large datasets  
- Connection pooling handled by Go's sql package
- CORS enabled for web application integration

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error

## Security Considerations

- Input validation on all parameters
- SQL injection protection via prepared statements
- CORS configured for cross-origin requests
- No authentication required (add as needed)

For production use, consider adding:
- Rate limiting
- Authentication/authorization
- HTTPS support
- Input sanitization
- Logging middleware