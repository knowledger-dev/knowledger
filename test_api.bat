@echo off
REM Base URL
SET BASE_URL=https://knowledger.onrender.com

REM Register a new user
curl -X POST "http://127.0.0.1:8000/register" ^
     -H "Content-Type: application/json" ^
     -d "{ \"username\": \"tester\", \"email\": \"tester@example.com\", \"password\": \"tester\" }"
echo Registered

REM Log in to obtain access token
curl -X POST "http://127.0.0.1:8000/token" ^
     -H "Content-Type: application/x-www-form-urlencoded" ^
     -d "username=tester&password=tester"
echo.
echo Please copy the access token from the response and paste it below.
SET /P TOKEN=Access Token:

REM Create a note
SET TIMESTAMP=2023-10-04T12:34:56Z
curl -X POST "https://knowledger.onrender.com/notes" ^
     -H "Content-Type: application/json" ^
     -H "Authorization: Bearer %TOKEN%" ^
     -d "{ \"content\": \"This is a test note for testing purposes.\", \"timestamp\": \"%TIMESTAMP%\" }"

REM Set the Note ID
echo.
echo Please copy the note ID from the response and paste it below.
SET /P NOTE_ID=Note ID:

REM Retrieve the created note
curl -X GET "%BASE_URL%/notes/%NOTE_ID%" ^
     -H "Authorization: Bearer %TOKEN%"

REM Query notes
curl -X POST "%BASE_URL%/query" ^
     -H "Content-Type: application/json" ^
     -H "Authorization: Bearer %TOKEN%" ^
     -d "{ \"query\": \"test note\", \"limit\": 5 }"

REM Perform RAG query
curl -X POST "%BASE_URL%/rag_query" ^
     -H "Content-Type: application/json" ^
     -H "Authorization: Bearer %TOKEN%" ^
     -d "{ \"query\": \"Explain the test note.\" }"

REM Trigger PageRank computation
curl -X POST "%BASE_URL%/compute_pagerank" ^
     -H "Authorization: Bearer %TOKEN%"

REM Trigger full recalculation and clustering
curl -X POST "%BASE_URL%/recalculate_all" ^
     -H "Authorization: Bearer %TOKEN%"

REM Test health endpoint
curl -X GET "%BASE_URL%/health"

REM Test MongoDB connection
curl -X GET "%BASE_URL%/test/mongodb"

REM Test LLM connection
curl -X GET "%BASE_URL%/test/llm"

REM Test model loading
curl -X GET "%BASE_URL%/test/model"
