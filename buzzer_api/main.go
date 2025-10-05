package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	_ "github.com/mattn/go-sqlite3"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// Config holds all configuration values
type Config struct {
	// Database Configuration
	DatabaseType     string
	DatabasePath     string
	DatabaseHost     string
	DatabasePort     int
	DatabaseName     string
	DatabaseUser     string
	DatabasePassword string

	// MySQL Configuration
	MySQLHost     string
	MySQLPort     int
	MySQLUser     string
	MySQLPassword string
	MySQLDatabase string

	// Server Configuration
	ServerHost string
	ServerPort int
	GinMode    string

	// API Configuration
	APIVersion       string
	CORSAllowOrigins string
	CORSAllowMethods string
	CORSAllowHeaders string

	// Pagination Configuration
	DefaultPageSize int
	MaxPageSize     int

	// Security Configuration
	JWTSecret       string
	APIRateLimit    int
	RateLimitWindow int

	// Logging Configuration
	LogLevel  string
	LogFormat string
}

// LoadConfig loads configuration from environment variables
func LoadConfig() (*Config, error) {
	// Try to load .env file from current directory and parent directories
	envPaths := []string{".env", "../.env", "../../.env"}

	for _, path := range envPaths {
		if _, err := os.Stat(path); err == nil {
			if err := godotenv.Load(path); err != nil {
				log.Printf("[CONFIG] Error loading %s: %v", path, err)
			} else {
				log.Printf("[CONFIG] Loaded configuration from: %s", path)
				break
			}
		}
	}

	config := &Config{
		// Database Configuration
		DatabaseType:     getEnv("DATABASE_TYPE", "sqlite3"),
		DatabasePath:     getEnv("DATABASE_PATH", "buzzer_detection.db"),
		DatabaseHost:     getEnv("DATABASE_HOST", "localhost"),
		DatabasePort:     getEnvInt("DATABASE_PORT", 5432),
		DatabaseName:     getEnv("DATABASE_NAME", "buzzer_detection"),
		DatabaseUser:     getEnv("DATABASE_USER", "postgres"),
		DatabasePassword: getEnv("DATABASE_PASSWORD", "password"),

		// MySQL Configuration
		MySQLHost:     getEnv("MYSQL_HOST", "localhost"),
		MySQLPort:     getEnvInt("MYSQL_PORT", 3306),
		MySQLUser:     getEnv("MYSQL_USER", "root"),
		MySQLPassword: getEnv("MYSQL_PASSWORD", "password"),
		MySQLDatabase: getEnv("MYSQL_DATABASE", "buzzer_detection"),

		// Server Configuration
		ServerHost: getEnv("SERVER_HOST", "0.0.0.0"),
		ServerPort: getEnvInt("SERVER_PORT", 8080),
		GinMode:    getEnv("GIN_MODE", "debug"),

		// API Configuration
		APIVersion:       getEnv("API_VERSION", "v1"),
		CORSAllowOrigins: getEnv("CORS_ALLOW_ORIGINS", "*"),
		CORSAllowMethods: getEnv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS"),
		CORSAllowHeaders: getEnv("CORS_ALLOW_HEADERS", "Content-Type,Authorization"),

		// Pagination Configuration
		DefaultPageSize: getEnvInt("DEFAULT_PAGE_SIZE", 20),
		MaxPageSize:     getEnvInt("MAX_PAGE_SIZE", 100),

		// Security Configuration
		JWTSecret:       getEnv("JWT_SECRET", "your-secret-key-here-change-in-production"),
		APIRateLimit:    getEnvInt("API_RATE_LIMIT", 100),
		RateLimitWindow: getEnvInt("RATE_LIMIT_WINDOW", 60),

		// Logging Configuration
		LogLevel:  strings.ToUpper(getEnv("LOG_LEVEL", "info")),
		LogFormat: getEnv("LOG_FORMAT", "json"),
	}

	return config, nil
}

// getEnv gets an environment variable with a default value
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvInt gets an environment variable as integer with a default value
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
		log.Printf("[CONFIG] Warning: Cannot convert %s='%s' to int, using default: %d", key, value, defaultValue)
	}
	return defaultValue
}

// getEnvBool gets an environment variable as boolean with a default value
func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		lowerValue := strings.ToLower(value)
		return lowerValue == "true" || lowerValue == "1" || lowerValue == "yes" || lowerValue == "on"
	}
	return defaultValue
}

// GetDatabaseConnectionString returns the appropriate database connection string
func (c *Config) GetDatabaseConnectionString() (string, string) {
	if strings.ToLower(c.DatabaseType) == "mysql" {
		// MySQL connection string
		dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=utf8mb4&parseTime=True&loc=Local",
			c.MySQLUser, c.MySQLPassword, c.MySQLHost, c.MySQLPort, c.MySQLDatabase)
		return "mysql", dsn
	} else {
		// SQLite connection string
		return "sqlite3", c.DatabasePath
	}
}

// GetServerAddress returns the server address
func (c *Config) GetServerAddress() string {
	return fmt.Sprintf("%s:%d", c.ServerHost, c.ServerPort)
}

// BuzzerAccount represents a buzzer account in the database with AI analysis
type BuzzerAccount struct {
	ID                  int     `json:"id" db:"id"`
	Username            string  `json:"username" db:"username"`
	DisplayName         *string `json:"display_name" db:"display_name"`
	SocialMediaPlatform string  `json:"social_media_platform" db:"social_media_platform"`
	AccountURL          string  `json:"account_url" db:"account_url"`
	PostsCount          int     `json:"posts_count" db:"posts_count"`
	FollowersCount      int     `json:"followers_count" db:"followers_count"`
	FollowingCount      int     `json:"following_count" db:"following_count"`
	AccountAgeDays      int     `json:"account_age_days" db:"account_age_days"`
	FreqPerDay          float64 `json:"freq_per_day" db:"freq_per_day"`
	Burstiness          float64 `json:"burstiness" db:"burstiness"`
	HashtagRatio        float64 `json:"hashtag_ratio" db:"hashtag_ratio"`
	URLRatio            float64 `json:"url_ratio" db:"url_ratio"`
	BuzzerProb          float64 `json:"buzzer_prob" db:"buzzer_prob"`
	BuzzerProbEnhanced  float64 `json:"buzzer_prob_enhanced" db:"buzzer_prob_enhanced"`
	RiskCategory        string  `json:"risk_category" db:"risk_category"`
	CoordCluster        int     `json:"coord_cluster" db:"coord_cluster"`
	CoordinationScore   float64 `json:"coordination_score" db:"coordination_score"`
	NetworkCentrality   float64 `json:"network_centrality" db:"network_centrality"`
	// Buzzer Type Classification
	BuzzerTypePrimary   string  `json:"buzzer_type_primary" db:"buzzer_type_primary"`
	BuzzerTypeSecondary *string `json:"buzzer_type_secondary" db:"buzzer_type_secondary"`
	ActivityPattern     string  `json:"activity_pattern" db:"activity_pattern"`
	TypeConfidence      float64 `json:"type_confidence" db:"type_confidence"`
	GovernmentScore     int     `json:"government_score" db:"government_score"`
	OppositionScore     int     `json:"opposition_score" db:"opposition_score"`
	CommercialScore     int     `json:"commercial_score" db:"commercial_score"`
	SpamScore           int     `json:"spam_score" db:"spam_score"`
	// AI Analysis Scores
	BehavioralAnomalyScore float64 `json:"behavioral_anomaly_score" db:"behavioral_anomaly_score"`
	TemporalAnomalyScore   float64 `json:"temporal_anomaly_score" db:"temporal_anomaly_score"`
	EngagementAnomalyScore float64 `json:"engagement_anomaly_score" db:"engagement_anomaly_score"`
	NetworkAnomalyScore    float64 `json:"network_anomaly_score" db:"network_anomaly_score"`
	ContentAnomalyScore    float64 `json:"content_anomaly_score" db:"content_anomaly_score"`
	AIConfidence           float64 `json:"ai_confidence" db:"ai_confidence"`
	// Content Analysis
	VocabularyDiversity float64 `json:"vocabulary_diversity" db:"vocabulary_diversity"`
	AvgSentenceLength   float64 `json:"avg_sentence_length" db:"avg_sentence_length"`
	GrammarComplexity   float64 `json:"grammar_complexity" db:"grammar_complexity"`
	ContentRepetition   float64 `json:"content_repetition" db:"content_repetition"`
	EmojiUsage          float64 `json:"emoji_usage" db:"emoji_usage"`
	CapsUsage           float64 `json:"caps_usage" db:"caps_usage"`
	// Sentiment Analysis
	SentimentPositive float64 `json:"sentiment_positive" db:"sentiment_positive"`
	SentimentNegative float64 `json:"sentiment_negative" db:"sentiment_negative"`
	SentimentNeutral  float64 `json:"sentiment_neutral" db:"sentiment_neutral"`
	// Writing Style
	WritingFormal     float64 `json:"writing_formal" db:"writing_formal"`
	WritingEmotional  float64 `json:"writing_emotional" db:"writing_emotional"`
	WritingPersuasive float64 `json:"writing_persuasive" db:"writing_persuasive"`
	// Timestamps
	DetectedAt time.Time `json:"detected_at" db:"detected_at"`
	QueryUsed  string    `json:"query_used" db:"query_used"`
	CreatedAt  time.Time `json:"created_at" db:"created_at"`
	UpdatedAt  time.Time `json:"updated_at" db:"updated_at"`
}

// Post represents a social media post
type Post struct {
	ID                  int       `json:"id" db:"id"`
	PostID              string    `json:"post_id" db:"post_id"`
	Username            string    `json:"username" db:"username"`
	SocialMediaPlatform string    `json:"social_media_platform" db:"social_media_platform"`
	Content             string    `json:"content" db:"content"`
	PostURL             string    `json:"post_url" db:"post_url"`
	PostDate            time.Time `json:"post_date" db:"post_date"`
	RetweetCount        int       `json:"retweet_count" db:"retweet_count"`
	LikeCount           int       `json:"like_count" db:"like_count"`
	ReplyCount          int       `json:"reply_count" db:"reply_count"`
	QuoteCount          int       `json:"quote_count" db:"quote_count"`
	IsRetweet           bool      `json:"is_retweet" db:"is_retweet"`
	OriginalAuthor      *string   `json:"original_author" db:"original_author"`
	QueryUsed           string    `json:"query_used" db:"query_used"`
	CreatedAt           time.Time `json:"created_at" db:"created_at"`
}

// CoordinationGroup represents a coordination group
type CoordinationGroup struct {
	ID                  int       `json:"id" db:"id"`
	GroupID             int       `json:"group_id" db:"group_id"`
	Username            string    `json:"username" db:"username"`
	SocialMediaPlatform string    `json:"social_media_platform" db:"social_media_platform"`
	SimilarityScore     float64   `json:"similarity_score" db:"similarity_score"`
	QueryUsed           string    `json:"query_used" db:"query_used"`
	DetectedAt          time.Time `json:"detected_at" db:"detected_at"`
}

// SearchResponse represents the response for search endpoints
type SearchResponse struct {
	Data       interface{} `json:"data"`
	Total      int         `json:"total"`
	Page       int         `json:"page"`
	PageSize   int         `json:"page_size"`
	TotalPages int         `json:"total_pages"`
}

// AIAnalysisDetails represents detailed AI analysis data
type AIAnalysisDetails struct {
	ID              int       `json:"id" db:"id"`
	Username        string    `json:"username" db:"username"`
	Platform        string    `json:"social_media_platform" db:"social_media_platform"`
	AnalysisType    string    `json:"analysis_type" db:"analysis_type"`
	AnalysisData    string    `json:"analysis_data" db:"analysis_data"` // JSON format
	ConfidenceScore float64   `json:"confidence_score" db:"confidence_score"`
	QueryUsed       string    `json:"query_used" db:"query_used"`
	CreatedAt       time.Time `json:"created_at" db:"created_at"`
}

// PostAnalysis represents post-level content analysis
type PostAnalysis struct {
	ID                  int       `json:"id" db:"id"`
	PostID              string    `json:"post_id" db:"post_id"`
	Username            string    `json:"username" db:"username"`
	Platform            string    `json:"social_media_platform" db:"social_media_platform"`
	ContentHash         string    `json:"content_hash" db:"content_hash"`
	SentimentScore      float64   `json:"sentiment_score" db:"sentiment_score"`
	EmotionScore        float64   `json:"emotion_score" db:"emotion_score"`
	SophisticationScore float64   `json:"sophistication_score" db:"sophistication_score"`
	SpamIndicators      float64   `json:"spam_indicators" db:"spam_indicators"`
	PromotionalScore    float64   `json:"promotional_score" db:"promotional_score"`
	AnalysisData        *string   `json:"analysis_data" db:"analysis_data"` // JSON format
	QueryUsed           string    `json:"query_used" db:"query_used"`
	CreatedAt           time.Time `json:"created_at" db:"created_at"`
}

// BuzzerTypeStats represents buzzer type distribution
type BuzzerTypeStats struct {
	BuzzerType        string  `json:"buzzer_type"`
	Count             int     `json:"count"`
	AverageProb       float64 `json:"average_probability"`
	AverageConfidence float64 `json:"average_confidence"`
}

// ActivityPatternStats represents activity pattern distribution
type ActivityPatternStats struct {
	Pattern      string  `json:"pattern"`
	Count        int     `json:"count"`
	AverageScore float64 `json:"average_score"`
}

// StatsResponse represents overall statistics with AI analysis
type StatsResponse struct {
	TotalAccounts        int                    `json:"total_accounts"`
	TotalPosts           int                    `json:"total_posts"`
	RiskDistribution     map[string]int         `json:"risk_distribution"`
	PlatformStats        map[string]int         `json:"platform_stats"`
	BuzzerTypeStats      []BuzzerTypeStats      `json:"buzzer_type_stats"`
	ActivityPatternStats []ActivityPatternStats `json:"activity_pattern_stats"`
	CoordinationGroups   int                    `json:"coordination_groups"`
	AIAnalysisEnabled    bool                   `json:"ai_analysis_enabled"`
	LastUpdate           time.Time              `json:"last_update"`
}

var (
	db     *sql.DB
	config *Config
)

func main() {
	// Load configuration
	var err error
	config, err = LoadConfig()
	if err != nil {
		log.Fatal("Failed to load configuration:", err)
	}

	// Set Gin mode
	gin.SetMode(config.GinMode)

	// Initialize database connection
	driver, dsn := config.GetDatabaseConnectionString()
	db, err = sql.Open(driver, dsn)
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()

	// Test database connection
	if err := db.Ping(); err != nil {
		log.Fatal("Failed to connect to database:", err)
	}
	log.Printf("[DB] Connected to %s database", config.DatabaseType)

	// Initialize Gin router
	r := gin.Default()

	// Add CORS middleware
	r.Use(func(c *gin.Context) {
		origins := strings.Split(config.CORSAllowOrigins, ",")
		methods := config.CORSAllowMethods
		headers := config.CORSAllowHeaders

		if len(origins) == 1 && origins[0] == "*" {
			c.Header("Access-Control-Allow-Origin", "*")
		} else {
			origin := c.Request.Header.Get("Origin")
			for _, allowedOrigin := range origins {
				if strings.TrimSpace(allowedOrigin) == origin {
					c.Header("Access-Control-Allow-Origin", origin)
					break
				}
			}
		}

		c.Header("Access-Control-Allow-Methods", methods)
		c.Header("Access-Control-Allow-Headers", headers)

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// API Routes
	api := r.Group(fmt.Sprintf("/api/%s", config.APIVersion))
	{
		// Buzzer accounts endpoints
		api.GET("/accounts", searchBuzzerAccounts)
		api.GET("/accounts/:username", getBuzzerAccount)
		api.GET("/accounts/platform/:platform", getAccountsByPlatform)
		api.GET("/accounts/risk/:level", getAccountsByRisk)
		api.GET("/accounts/type/:type", getAccountsByType)
		api.GET("/accounts/pattern/:pattern", getAccountsByActivityPattern)

		// Posts endpoints
		api.GET("/posts", searchPosts)
		api.GET("/posts/user/:username", getPostsByUser)
		api.GET("/posts/platform/:platform", getPostsByPlatform)
		api.GET("/posts/analysis/:postId", getPostAnalysis)

		// AI Analysis endpoints
		api.GET("/ai/analysis/:username", getAIAnalysisDetails)
		api.GET("/ai/content-analysis", getContentAnalysisStats)
		api.GET("/ai/sentiment-analysis", getSentimentAnalysisStats)
		api.GET("/ai/behavioral-analysis", getBehavioralAnalysisStats)

		// Coordination endpoints
		api.GET("/coordination/groups", getCoordinationGroups)
		api.GET("/coordination/group/:groupId", getCoordinationGroup)

		// Statistics endpoints
		api.GET("/stats", getStats)
		api.GET("/stats/platforms", getPlatformStats)
		api.GET("/stats/risks", getRiskStats)
		api.GET("/stats/types", getBuzzerTypeStats)
		api.GET("/stats/patterns", getActivityPatternStats)
		api.GET("/stats/ai-metrics", getAIMetricsStats)

		// Search endpoints
		api.GET("/search", globalSearch)
		api.GET("/search/advanced", advancedSearch)
	}

	// Swagger documentation
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

	// Health check
	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":    "OK",
			"timestamp": time.Now(),
			"database":  config.DatabaseType,
			"version":   config.APIVersion,
		})
	})

	// Configuration endpoint
	r.GET("/config", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"api_version":       config.APIVersion,
			"database_type":     config.DatabaseType,
			"default_page_size": config.DefaultPageSize,
			"max_page_size":     config.MaxPageSize,
		})
	})

	// Start server
	serverAddr := config.GetServerAddress()
	log.Printf("Starting Buzzer Detection API on: http://%s", serverAddr)
	log.Printf("Swagger documentation: http://%s/swagger/index.html", serverAddr)
	log.Printf("Health check: http://%s/health", serverAddr)
	log.Printf("Configuration: http://%s/config", serverAddr)

	r.Run(serverAddr)
}
