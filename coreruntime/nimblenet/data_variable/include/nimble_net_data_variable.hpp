/*
 * SPDX-FileCopyrightText: (C) 2025 DeliteAI Authors
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <type_traits>

#include "dataframe_variable.hpp"
#include "model_nimble_net_variable.hpp"
#include "nimble_net_util.hpp"
#include "nlohmann/json_fwd.hpp"
#include "pre_processor_nimble_net_variable.hpp"
#include "raw_event_store_data_variable.hpp"
#include "single_variable.hpp"
#include "tensor_data_variable.hpp"

#ifndef MINIMAL_BUILD
#include "concurrent_executor_variable.hpp"
#endif  // MINIMAL_BUILD

class CommandCenter;

/**
 * @brief Main data variable class for NimbleNet operations
 *
 * NimbleNetDataVariable serves as the primary interface for all NimbleNet-specific
 * operations in the data variable system. It provides functionality for creating
 * tensors, loading models and LLMs, mathematical operations, data management,
 * and system configuration access.
 *
 * The class implements a comprehensive set of operations including:
 * - Tensor creation and manipulation
 * - Model and LLM loading with async support
 * - Mathematical functions (exp, pow, min, max, sum, mean, log)
 * - Data storage and retrieval (raw events, dataframes)
 * - System utilities (time, configuration access)
 * - Concurrent execution support
 *
 * All operations are dispatched through the call_function method using
 * member function indices, providing a unified interface for script execution.
 */
class NimbleNetDataVariable final : public DataVariable {
  CommandCenter* _commandCenter = nullptr; /**< Pointer to the command center for system operations */

  int get_containerType() const override { return CONTAINERTYPE::SINGLE; }

  bool get_bool() override { return true; }

  int get_dataType_enum() const override { return DATATYPE::NIMBLENET; }

  /*
  DELITEPY_DOC_BLOCK_BEGIN

def zeros(shape: list[int], dtype: str) -> Tensor:
    """
    Creates and return a tensor with zeroes of given shape and data type.

    Parameters
    ----------
    shape : list[int]
        Desired shape of the tensor.
    dtype : str
        Data type with which to create the tensor.

    Returns
    ----------
    tensor : Tensor
        Returns the tensor of the shape and data type filled with zeros.
    """
    pass
  DELITEPY_DOC_BLOCK_END
  */
  OpReturnType create_tensor(const std::vector<OpReturnType>& arguments);
  OpReturnType load_model(const std::vector<OpReturnType>& arguments, CallStack& stack);
  OpReturnType load_llm(const std::vector<OpReturnType>& arguments, CallStack& stack);

  OpReturnType get_current_time(const std::vector<OpReturnType>& arguments);

  OpReturnType get_config(const std::vector<OpReturnType>& arguments);

  OpReturnType get_exp(const std::vector<OpReturnType>& arguments);
  OpReturnType get_pow(const std::vector<OpReturnType>& arguments);

  OpReturnType get_raw_events_store(const std::vector<OpReturnType>& arguments);

  OpReturnType get_dataframe(const std::vector<OpReturnType>& arguments);

  OpReturnType min(const std::vector<OpReturnType>& args);

  OpReturnType max(const std::vector<OpReturnType>& args);

  OpReturnType sum(const std::vector<OpReturnType>& args);

  OpReturnType mean(const std::vector<OpReturnType>& args);

  OpReturnType log(const std::vector<OpReturnType>& args);

  OpReturnType create_retriever(const std::vector<OpReturnType>& arguments, CallStack& stack);

  OpReturnType create_json_document(const std::vector<OpReturnType>& arguments, CallStack& stack) {
    THROW("%s", "Currently not supporting loading JSON document directly");
  }

  std::vector<std::map<std::string, std::string>> get_compatible_llms(CommandCenter* commandCenter);

  OpReturnType list_compatible_llms(const std::vector<OpReturnType>& arguments);

  OpReturnType create_concurrent_executor(const std::vector<OpReturnType>& arguments);

  OpReturnType set_threads(const std::vector<OpReturnType>& arguments);

  // UTF-8 helper function to determine if byte is a continuation byte
    bool is_utf8_continuation(unsigned char byte)
    {
        return (byte & 0xC0) == 0x80;
    }

// UTF-8 helper function to get next character and advance pointer
    char32_t get_next_utf8_char(const char *&p)
    {
        const unsigned char *up = reinterpret_cast<const unsigned char *>(p);

        // Single byte (ASCII)
        if (up[0] < 0x80)
        {
            char32_t result = up[0];
            p++;
            return result;
        }

        // 2-byte sequence
        if ((up[0] & 0xE0) == 0xC0)
        {
            if ((up[1] & 0xC0) == 0x80)
            {
                char32_t result = ((up[0] & 0x1F) << 6) | (up[1] & 0x3F);
                p += 2;
                return result;
            }
        }

        // 3-byte sequence
        if ((up[0] & 0xF0) == 0xE0)
        {
            if ((up[1] & 0xC0) == 0x80 && (up[2] & 0xC0) == 0x80)
            {
                char32_t result = ((up[0] & 0x0F) << 12) | ((up[1] & 0x3F) << 6) | (up[2] & 0x3F);
                p += 3;
                return result;
            }
        }

        // 4-byte sequence
        if ((up[0] & 0xF8) == 0xF0)
        {
            if ((up[1] & 0xC0) == 0x80 && (up[2] & 0xC0) == 0x80 && (up[3] & 0xC0) == 0x80)
            {
                char32_t result = ((up[0] & 0x07) << 18) | ((up[1] & 0x3F) << 12) |
                                  ((up[2] & 0x3F) << 6) | (up[3] & 0x3F);
                p += 4;
                return result;
            }
        }

        // Invalid sequence, skip one byte
        p++;
        return 0xFFFD; // Unicode replacement character
    }

// Function to write a UTF-8 character to a buffer
    void append_utf8_char(char32_t ch, char *&buffer)
    {
        if (ch < 0x80)
        {
            // 1-byte sequence
            *buffer++ = static_cast<char>(ch);
        }
        else if (ch < 0x800)
        {
            // 2-byte sequence
            *buffer++ = static_cast<char>(0xC0 | (ch >> 6));
            *buffer++ = static_cast<char>(0x80 | (ch & 0x3F));
        }
        else if (ch < 0x10000)
        {
            // 3-byte sequence
            *buffer++ = static_cast<char>(0xE0 | (ch >> 12));
            *buffer++ = static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
            *buffer++ = static_cast<char>(0x80 | (ch & 0x3F));
        }
        else if (ch < 0x110000)
        {
            // 4-byte sequence
            *buffer++ = static_cast<char>(0xF0 | (ch >> 18));
            *buffer++ = static_cast<char>(0x80 | ((ch >> 12) & 0x3F));
            *buffer++ = static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
            *buffer++ = static_cast<char>(0x80 | (ch & 0x3F));
        }
    }

// Function to check if a Unicode character is a stress marker
    bool is_stress_marker(char32_t ch)
    {
        // Unicode code points for stress markers
        return ch == 0x02C8 || // U+02C8 MODIFIER LETTER VERTICAL LINE (ˈ)
               ch == 0x02CC || ch == '_';   // U+02CC MODIFIER LETTER LOW VERTICAL LINE (ˌ)

        // Add any other stress markers here if needed
    }

// UTF-8 aware string replacement function
    std::string replace_substring(const std::string &str, const std::string &from, const std::string &to)
    {
        if (from.empty())
            return str;

        std::string result;
        result.reserve(str.length()); // Pre-allocate memory to avoid reallocations

        size_t pos = 0;
        size_t lastPos = 0;

        // Use a searching algorithm that respects UTF-8 character boundaries
        while ((pos = str.find(from, lastPos)) != std::string::npos)
        {
            // Verify we're at a character boundary by checking the byte isn't a UTF-8 continuation byte
            bool validBoundary = true;
            if (pos > 0 && is_utf8_continuation(static_cast<unsigned char>(str[pos])))
            {
                validBoundary = false;
            }

            if (validBoundary)
            {
                // Append characters from lastPos to pos
                result.append(str, lastPos, pos - lastPos);
                // Append the replacement
                result.append(to);
                lastPos = pos + from.length();
            }
            else
            {
                // This is a false match (within a multi-byte character) - skip one byte forward
                pos++;
                continue;
            }
        }

        // Append any remaining characters
        result.append(str, lastPos, std::string::npos);
        return result;
    }

// Alternative UTF-8 aware string replacement function using our character-by-character parsing
    std::string replace_substring_robust(const std::string &str, const std::string &from, const std::string &to)
    {
        if (from.empty())
            return str;

        std::string result;
        result.reserve(str.length()); // Reserve space for efficiency

        // Check if we're processing multi-byte characters
        bool hasMultibyteChars = false;
        for (unsigned char c : from)
        {
            if (c >= 0x80)
            {
                hasMultibyteChars = true;
                break;
            }
        }

        // For ASCII-only patterns, we can use the standard method
        if (!hasMultibyteChars)
        {
            return replace_substring(str, from, to);
        }

        // For patterns with multi-byte characters, use character-by-character comparison
        const char *strData = str.c_str();
        size_t strPos = 0;

        while (strData[strPos])
        {
            if (str.compare(strPos, from.length(), from) == 0)
            {
                // Found a match
                result.append(to);
                strPos += from.length();
            }
            else
            {
                // No match, copy the current character
                const char *current = strData + strPos;
                char32_t ch = get_next_utf8_char(current);

                // Append the original bytes for this character
                result.append(strData + strPos, current - (strData + strPos));
                strPos = current - strData;
            }
        }

        return result;
    }

// Function to apply phoneme transformations for eSpeak output
    std::string transform_phonemes(const std::string &phonemes)
    {
        // Define the E2M replacements
        const std::vector<std::pair<std::string, std::string>> E2M = {
                {"a^ɪ", "I"},
                {"a^ʊ", "W"},
                {"d^z", "ʣ"},
                {"d^ʒ", "ʤ"},
                {"e^ɪ", "A"},
                {"o^ʊ", "O"},
                {"s^s", "S"},
                {"t^s", "ʦ"},
                {"t^ʃ", "ʧ"},
                {"ɔ^ɪ", "Y"},
                {"ə^ʊ", "Q"},
                {"ɜːɹ", "ɜɹ"},
                {"ɔː", "ɔɹ"},
                {"ɪə", "iə"},
                {"^", ""},
                {"and", "ænd"},
                {":",""}
        };

        // Apply E2M replacements
        std::string result = phonemes;
        for (const auto &pair : E2M)
        {
            result = replace_substring_robust(result, pair.first, pair.second);
        }
        return result;
    }

// Combined function to remove stress markers and apply phoneme transformations
    std::string process_phonemes(const char *phonemes)
    {
        if (!phonemes)
            return "";

        // First, remove stress markers
        std::string without_stress;
        const char *src = phonemes;

        while (*src != '\0')
        {
            const char *current = src;
            char32_t ch = get_next_utf8_char(src);

            if (!is_stress_marker(ch))
            {
                // Append the original bytes for this character
                without_stress.append(current, src - current);
            }
        }

        // Then apply phoneme transformations
        return transform_phonemes(without_stress);
    }

  OpReturnType convertTextToPhonemes(const std::vector<OpReturnType>& arguments) override {
    THROW_ARGUMENTS_NOT_MATCH(arguments.size(), 1, MemberFuncType::CONVERT_TEXT_TO_PHONEMES);
    std::string text = arguments[0]->get_string();
    // phonemes = espeak_TextToPhonemes(&pText, espeakCHARS_UTF8, 24322);
    const char *pText = text.c_str();
    const char *phonemes = nullptr;
    std::string phonemesStr;

    #if defined(__ANDROID__)

    #elif defined(IOS)
        phonemes = get_phonemes(pText);
        phonemesStr = process_phonemes(phonemes);
        if (phonemes) {
            free((void*)phonemes);
        }
    #else
        THROW("espeak only supported in android and ios");
    #endif
    return std::make_shared<SingleVariable<std::string>>(phonemesStr);
  }
  OpReturnType initializeEspeak() override{
      std::string homeDirectory = nativeinterface::HOMEDIR;
          auto path = homeDirectory.c_str();
          int sampleRate;
          #if defined(__ANDROID__)

          #elif defined(IOS)
            sampleRate = initialize_espeak(path);
          #else
          #endif
          return std::make_shared<SingleVariable<std::int64_t>>(sampleRate);
    }

  OpReturnType call_function(int memberFuncIndex, const std::vector<OpReturnType>& arguments,
                             CallStack& stack) override;

  nlohmann::json to_json() const override { return "[NimbleNet]"; }

 public:
  NimbleNetDataVariable(CommandCenter* commandCenter) { _commandCenter = commandCenter; }

  std::string print() override { return fallback_print(); }
};
