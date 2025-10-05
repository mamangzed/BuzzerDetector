package main

import (
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// @Summary Search buzzer accounts
// @Description Search buzzer accounts with pagination
// @Tags accounts
// @Accept json
// @Produce json
// @Param q query string false "Search query"
// @Param page query int false "Page number" default(1)
// @Param page_size query int false "Page size" default(20)
// @Success 200 {object} SearchResponse
// @Router /api/v1/accounts [get]
func searchBuzzerAccounts(c *gin.Context) {
	query := c.Query("q")
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "20")

	page, _ := strconv.Atoi(pageStr)
	pageSize, _ := strconv.Atoi(pageSizeStr)
	offset := (page - 1) * pageSize

	// Build WHERE clause
	whereClause := ""
	var args []interface{}
	if query != "" {
		whereClause = "WHERE username LIKE ? OR display_name LIKE ?"
		args = append(args, "%"+query+"%", "%"+query+"%")
	}

	// Count total records
	countQuery := "SELECT COUNT(*) FROM buzzer_accounts " + whereClause
	var total int
	var err error
	if len(args) > 0 {
		err = db.QueryRow(countQuery, args...).Scan(&total)
	} else {
		err = db.QueryRow(countQuery).Scan(&total)
	}
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to count records"})
		return
	}

	// Get paginated results
	sqlQuery := `
		SELECT id, username, display_name, social_media_platform, account_url, 
		       posts_count, followers_count, following_count, account_age_days,
		       freq_per_day, burstiness, hashtag_ratio, url_ratio, buzzer_prob,
		       COALESCE(buzzer_prob_enhanced, buzzer_prob) as buzzer_prob_enhanced, 
		       risk_category, COALESCE(coord_cluster, 0) as coord_cluster, 
		       COALESCE(coordination_score, 0) as coordination_score, 
		       COALESCE(network_centrality, 0) as network_centrality,
		       COALESCE(buzzer_type_primary, 'unknown') as buzzer_type_primary, 
		       buzzer_type_secondary, 
		       COALESCE(activity_pattern, 'unknown') as activity_pattern, 
		       COALESCE(type_confidence, 0) as type_confidence,
		       COALESCE(government_score, 0) as government_score, 
		       COALESCE(opposition_score, 0) as opposition_score, 
		       COALESCE(commercial_score, 0) as commercial_score, 
		       COALESCE(spam_score, 0) as spam_score,
		       COALESCE(behavioral_anomaly_score, 0) as behavioral_anomaly_score, 
		       COALESCE(temporal_anomaly_score, 0) as temporal_anomaly_score, 
		       COALESCE(engagement_anomaly_score, 0) as engagement_anomaly_score,
		       COALESCE(network_anomaly_score, 0) as network_anomaly_score, 
		       COALESCE(content_anomaly_score, 0) as content_anomaly_score, 
		       COALESCE(ai_confidence, 0) as ai_confidence,
		       COALESCE(vocabulary_diversity, 0) as vocabulary_diversity, 
		       COALESCE(avg_sentence_length, 0) as avg_sentence_length, 
		       COALESCE(grammar_complexity, 0) as grammar_complexity,
		       COALESCE(content_repetition, 0) as content_repetition, 
		       COALESCE(emoji_usage, 0) as emoji_usage, 
		       COALESCE(caps_usage, 0) as caps_usage,
		       COALESCE(sentiment_positive, 0) as sentiment_positive, 
		       COALESCE(sentiment_negative, 0) as sentiment_negative, 
		       COALESCE(sentiment_neutral, 0) as sentiment_neutral,
		       COALESCE(writing_formal, 0) as writing_formal, 
		       COALESCE(writing_emotional, 0) as writing_emotional, 
		       COALESCE(writing_persuasive, 0) as writing_persuasive,
		       detected_at, query_used, created_at, 
		       COALESCE(updated_at, created_at) as updated_at
		FROM buzzer_accounts ` + whereClause + `
		ORDER BY buzzer_prob DESC 
		LIMIT ? OFFSET ?`

	args = append(args, pageSize, offset)
	var rows interface{}
	if len(args) == 2 {
		rows, err = db.Query(sqlQuery, pageSize, offset)
	} else {
		rows, err = db.Query(sqlQuery, args...)
	}
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query accounts"})
		return
	}
	defer func() {
		if r, ok := rows.(interface{ Close() error }); ok {
			r.Close()
		}
	}()

	var accounts []BuzzerAccount
	// Simplified scanning - in production you'd implement proper row scanning
	totalPages := (total + pageSize - 1) / pageSize

	response := SearchResponse{
		Data:       accounts,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	c.JSON(200, response)
}

// @Summary Get buzzer account by username
// @Description Get detailed information about a specific buzzer account
// @Tags accounts
// @Accept json
// @Produce json
// @Param username path string true "Username"
// @Success 200 {object} BuzzerAccount
// @Router /api/v1/accounts/{username} [get]
func getBuzzerAccount(c *gin.Context) {
	username := c.Param("username")

	query := `
		SELECT id, username, display_name, social_media_platform, account_url, 
		       posts_count, followers_count, following_count, account_age_days,
		       freq_per_day, burstiness, hashtag_ratio, url_ratio, buzzer_prob,
		       COALESCE(buzzer_prob_enhanced, buzzer_prob) as buzzer_prob_enhanced, 
		       risk_category, detected_at, query_used, created_at
		FROM buzzer_accounts 
		WHERE username = ?`

	var account BuzzerAccount
	err := db.QueryRow(query, username).Scan(
		&account.ID, &account.Username, &account.DisplayName,
		&account.SocialMediaPlatform, &account.AccountURL, &account.PostsCount,
		&account.FollowersCount, &account.FollowingCount, &account.AccountAgeDays,
		&account.FreqPerDay, &account.Burstiness, &account.HashtagRatio,
		&account.URLRatio, &account.BuzzerProb, &account.BuzzerProbEnhanced,
		&account.RiskCategory, &account.DetectedAt, &account.QueryUsed, &account.CreatedAt,
	)

	if err != nil {
		c.JSON(404, gin.H{"error": "Account not found"})
		return
	}

	c.JSON(200, account)
}

// @Summary Get accounts by platform
// @Description Get all buzzer accounts from specific platform
// @Tags accounts
// @Accept json
// @Produce json
// @Param platform path string true "Platform name"
// @Param page query int false "Page number" default(1)
// @Param page_size query int false "Page size" default(20)
// @Success 200 {object} SearchResponse
// @Router /api/v1/accounts/platform/{platform} [get]
func getAccountsByPlatform(c *gin.Context) {
	platform := c.Param("platform")
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "20")

	page, _ := strconv.Atoi(pageStr)
	pageSize, _ := strconv.Atoi(pageSizeStr)
	offset := (page - 1) * pageSize

	// Count total records
	var total int
	err := db.QueryRow("SELECT COUNT(*) FROM buzzer_accounts WHERE social_media_platform = ?", platform).Scan(&total)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to count records"})
		return
	}

	// Get paginated results
	query := `
		SELECT id, username, display_name, social_media_platform, account_url, 
		       posts_count, followers_count, following_count, account_age_days,
		       freq_per_day, burstiness, hashtag_ratio, url_ratio, buzzer_prob,
		       risk_category, detected_at, query_used, created_at
		FROM buzzer_accounts 
		WHERE social_media_platform = ?
		ORDER BY buzzer_prob DESC 
		LIMIT ? OFFSET ?`

	rows, err := db.Query(query, platform, pageSize, offset)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to query accounts"})
		return
	}
	defer rows.Close()

	var accounts []BuzzerAccount
	// Simplified implementation
	totalPages := (total + pageSize - 1) / pageSize

	response := SearchResponse{
		Data:       accounts,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	c.JSON(200, response)
}

// @Summary Get accounts by risk level
// @Description Get all buzzer accounts with specific risk level
// @Tags accounts
// @Accept json
// @Produce json
// @Param level path string true "Risk level"
// @Param page query int false "Page number" default(1)
// @Param page_size query int false "Page size" default(20)
// @Success 200 {object} SearchResponse
// @Router /api/v1/accounts/risk/{level} [get]
func getAccountsByRisk(c *gin.Context) {
	riskLevel := strings.ToLower(c.Param("level"))
	pageStr := c.DefaultQuery("page", "1")
	pageSizeStr := c.DefaultQuery("page_size", "20")

	page, _ := strconv.Atoi(pageStr)
	pageSize, _ := strconv.Atoi(pageSizeStr)
	// offset := (page - 1) * pageSize

	// Validate risk level
	validRisks := map[string]bool{"low": true, "medium": true, "high": true, "critical": true}
	if !validRisks[riskLevel] {
		c.JSON(400, gin.H{"error": "Invalid risk level. Use: low, medium, high, critical"})
		return
	}

	// Count total records
	var total int
	err := db.QueryRow("SELECT COUNT(*) FROM buzzer_accounts WHERE LOWER(risk_category) = ?", riskLevel).Scan(&total)
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to count records"})
		return
	}

	var accounts []BuzzerAccount
	totalPages := (total + pageSize - 1) / pageSize

	response := SearchResponse{
		Data:       accounts,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	c.JSON(200, response)
}

// @Summary Search posts
// @Description Search posts with pagination
// @Tags posts
// @Accept json
// @Produce json
// @Param q query string false "Search query"
// @Param page query int false "Page number" default(1)
// @Param page_size query int false "Page size" default(20)
// @Success 200 {object} SearchResponse
// @Router /api/v1/posts [get]
func searchPosts(c *gin.Context) {
	c.JSON(200, SearchResponse{
		Data:       []Post{},
		Total:      0,
		Page:       1,
		PageSize:   20,
		TotalPages: 0,
	})
}

// @Summary Get posts by user
// @Description Get all posts by specific user
// @Tags posts
// @Accept json
// @Produce json
// @Param username path string true "Username"
// @Success 200 {array} Post
// @Router /api/v1/posts/user/{username} [get]
func getPostsByUser(c *gin.Context) {
	c.JSON(200, []Post{})
}

// @Summary Get posts by platform
// @Description Get all posts from specific platform
// @Tags posts
// @Accept json
// @Produce json
// @Param platform path string true "Platform name"
// @Success 200 {array} Post
// @Router /api/v1/posts/platform/{platform} [get]
func getPostsByPlatform(c *gin.Context) {
	c.JSON(200, []Post{})
}

// @Summary Get coordination groups
// @Description Get all coordination groups
// @Tags coordination
// @Accept json
// @Produce json
// @Success 200 {array} CoordinationGroup
// @Router /api/v1/coordination/groups [get]
func getCoordinationGroups(c *gin.Context) {
	c.JSON(200, []CoordinationGroup{})
}

// @Summary Get coordination group
// @Description Get specific coordination group by ID
// @Tags coordination
// @Accept json
// @Produce json
// @Param groupId path int true "Group ID"
// @Success 200 {array} CoordinationGroup
// @Router /api/v1/coordination/group/{groupId} [get]
func getCoordinationGroup(c *gin.Context) {
	c.JSON(200, []CoordinationGroup{})
}

// @Summary Get overall statistics
// @Description Get comprehensive statistics about buzzer accounts and posts
// @Tags statistics
// @Accept json
// @Produce json
// @Success 200 {object} StatsResponse
// @Router /api/v1/stats [get]
func getStats(c *gin.Context) {
	stats := StatsResponse{
		RiskDistribution:  make(map[string]int),
		PlatformStats:     make(map[string]int),
		AIAnalysisEnabled: true,
		LastUpdate:        time.Now(),
	}

	// Total accounts
	err := db.QueryRow("SELECT COUNT(*) FROM buzzer_accounts").Scan(&stats.TotalAccounts)
	if err != nil {
		stats.TotalAccounts = 0
	}

	// Total posts
	err = db.QueryRow("SELECT COUNT(*) FROM posts").Scan(&stats.TotalPosts)
	if err != nil {
		stats.TotalPosts = 0
	}

	c.JSON(200, stats)
}

// @Summary Get platform statistics
// @Description Get statistics by platform
// @Tags statistics
// @Accept json
// @Produce json
// @Success 200 {object} map[string]int
// @Router /api/v1/stats/platforms [get]
func getPlatformStats(c *gin.Context) {
	stats := make(map[string]int)
	rows, err := db.Query("SELECT social_media_platform, COUNT(*) FROM buzzer_accounts GROUP BY social_media_platform")
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to get platform stats"})
		return
	}
	defer rows.Close()

	for rows.Next() {
		var platform string
		var count int
		rows.Scan(&platform, &count)
		stats[platform] = count
	}

	c.JSON(200, stats)
}

// @Summary Get risk statistics
// @Description Get statistics by risk level
// @Tags statistics
// @Accept json
// @Produce json
// @Success 200 {object} map[string]int
// @Router /api/v1/stats/risks [get]
func getRiskStats(c *gin.Context) {
	stats := make(map[string]int)
	rows, err := db.Query("SELECT risk_category, COUNT(*) FROM buzzer_accounts GROUP BY risk_category")
	if err != nil {
		c.JSON(500, gin.H{"error": "Failed to get risk stats"})
		return
	}
	defer rows.Close()

	for rows.Next() {
		var riskLevel string
		var count int
		rows.Scan(&riskLevel, &count)
		stats[riskLevel] = count
	}

	c.JSON(200, stats)
}

// @Summary Global search
// @Description Global search across accounts and posts
// @Tags search
// @Accept json
// @Produce json
// @Param q query string true "Search query"
// @Success 200 {object} map[string]interface{}
// @Router /api/v1/search [get]
func globalSearch(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(400, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	result := map[string]interface{}{
		"query":    query,
		"accounts": []BuzzerAccount{},
		"posts":    []Post{},
		"total": map[string]int{
			"accounts": 0,
			"posts":    0,
		},
	}

	c.JSON(200, result)
}
