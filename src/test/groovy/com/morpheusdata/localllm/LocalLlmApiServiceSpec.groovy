package com.morpheusdata.localllm

import spock.lang.Specification
import spock.lang.Unroll

class LocalLlmApiServiceSpec extends Specification {

	@Unroll
	def "parseBoolean('#value', #defaultValue) == #expected"() {
		expect:
		LocalLlmApiService.parseBoolean(value, defaultValue) == expected

		where:
		value   | defaultValue || expected
		null    | true         || true
		''      | true         || true
		'on'    | false        || true
		'true'  | false        || true
		true    | false        || true
		'off'   | true         || false
		'false' | true         || false
		false   | true         || false
		'junk'  | true         || true
	}

	def "resolveIgnoreSsl defaults to true when not specified"() {
		expect:
		LocalLlmApiService.resolveIgnoreSsl(null)
		LocalLlmApiService.resolveIgnoreSsl([:])
		!LocalLlmApiService.resolveIgnoreSsl([ignoreSSL: false])
		!LocalLlmApiService.resolveIgnoreSsl([ignoreSSL: 'off'])
	}

	def "buildRequestOptions passes ignoreSSL through"() {
		given:
		def service = new LocalLlmApiService()

		expect:
		service.buildRequestOptions('t', [:], null, 1000).ignoreSSL == true
		service.buildRequestOptions('t', [:], null, 1000, false).ignoreSSL == false
	}
}
