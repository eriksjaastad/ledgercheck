# Inputs. WRITE-ONLY reference stack: nothing in this repo applies it.
#
# Secrets are variables marked `sensitive` with a null default. Pass them at
# plan time from your secret store (TF_VAR_langfuse_secret_key=...), never in
# a committed file. terraform.tfvars.example lists the non-secret inputs.

variable "subscription_id" {
  description = "Your Azure subscription id."
  type        = string
}

variable "name" {
  description = "Name prefix for every resource."
  type        = string
  default     = "ledgercheck"
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "westus2"
}

variable "registry_server" {
  description = "Container registry login server, e.g. <acr>.azurecr.io."
  type        = string
}

variable "registry_id" {
  description = "Resource id of that registry; the job identity gets AcrPull on it."
  type        = string
}

variable "image" {
  description = "Image built from the repo Dockerfile, e.g. <acr>.azurecr.io/ledgercheck:<tag>."
  type        = string
}

variable "args" {
  description = "Arguments to the image's `ledgercheck` entrypoint."
  type        = list(string)
  default     = ["judge"]
}

variable "enable_langfuse" {
  description = "Wire LANGFUSE_* into the job. The image must be built with WITH_LANGFUSE=1."
  type        = bool
  default     = false
}

variable "langfuse_public_key" {
  description = "Langfuse public key (secret). Required when enable_langfuse."
  type        = string
  default     = null
  sensitive   = true
}

variable "langfuse_secret_key" {
  description = "Langfuse secret key (secret). Required when enable_langfuse."
  type        = string
  default     = null
  sensitive   = true
}

variable "langfuse_host" {
  description = "Langfuse host; null uses the SDK default."
  type        = string
  default     = null
}

variable "live_llm" {
  description = "Set LEDGERCHECK_LLM=1 (the spend gate). Off by default (live calls cost money)."
  type        = bool
  default     = false
}

variable "openrouter_api_key" {
  description = "OpenRouter API key (secret). Required when live_llm."
  type        = string
  default     = null
  sensitive   = true
}
